#!/usr/bin/env python3
"""Generate final benchmark report.

Merges all pipeline logs into a final catalog of validated
benchmark models and generates a markdown report with
funnel statistics and model metadata.

Usage:
    python step5_report.py
"""

import argparse
import logging
from pathlib import Path

import config
import pandas as pd
import utils

logger = logging.getLogger(__name__)


def build_benchmark_catalog() -> pd.DataFrame:
    """Build the final benchmark model catalog.

    Merges filter, conversion, and validation logs to
    produce one row per model that passed all pipeline
    stages.
    """
    # Load available logs
    candidates_path = Path(config.CANDIDATES_CSV)
    conversion_path = Path(config.CONVERSION_LOG_CSV)
    validation_path = Path(config.VALIDATION_LOG_CSV)

    if not validation_path.exists():
        logger.error(f"Validation log not found: {validation_path}")
        return pd.DataFrame()

    val_df = pd.read_csv(validation_path)

    # Start with validated models
    catalog = val_df[["model_id"]].copy()

    # Merge validation results
    catalog = catalog.merge(
        val_df[
            [
                "model_id",
                "load_ok",
                "ode_ok",
                "ssa_consistent",
                "ode_cross_ok",
                "ssa_cross_ok",
            ]
        ],
        on="model_id",
        how="left",
    )

    # Merge conversion info if available
    if conversion_path.exists():
        conv_df = pd.read_csv(conversion_path)
        catalog = catalog.merge(
            conv_df[
                [
                    "model_id",
                    "net_n_species",
                    "net_n_reactions",
                ]
            ],
            on="model_id",
            how="left",
        )

    # Merge filter info if available
    if candidates_path.exists():
        cand_df = pd.read_csv(candidates_path)
        catalog = catalog.merge(
            cand_df[
                [
                    "model_id",
                    "n_species",
                    "n_reactions",
                    "n_parameters",
                ]
            ],
            on="model_id",
            how="left",
        )

    # Add path columns
    net_dir = Path(config.NET_MODELS_DIR)
    sbml_dir = Path(config.SBML_CANDIDATES_DIR)
    catalog["net_path"] = catalog["model_id"].apply(lambda m: str(net_dir / f"{m}.net"))
    catalog["sbml_path"] = catalog["model_id"].apply(lambda m: str(sbml_dir / f"{m}.xml"))

    return catalog


def generate_funnel_stats() -> dict:
    """Compute funnel statistics across all stages."""
    stats = {}

    # Count SBML downloads
    sbml_dir = Path(config.SBML_DOWNLOADS_DIR)
    if sbml_dir.exists():
        stats["sbml_downloaded"] = len(list(sbml_dir.glob("*.xml")))
    else:
        stats["sbml_downloaded"] = 0

    # Count candidates
    cand_dir = Path(config.SBML_CANDIDATES_DIR)
    if cand_dir.exists():
        stats["candidates"] = len(list(cand_dir.glob("*.xml")))
    else:
        stats["candidates"] = 0

    # Count conversions
    conv_path = Path(config.CONVERSION_LOG_CSV)
    if conv_path.exists():
        conv_df = pd.read_csv(conv_path)
        if "atomize_success" in conv_df.columns:
            stats["atomized"] = int(conv_df["atomize_success"].sum())
        else:
            stats["atomized"] = 0
        stats["net_generated"] = int(conv_df["net_success"].sum())
    else:
        stats["atomized"] = 0
        stats["net_generated"] = 0

    # Count validations
    val_path = Path(config.VALIDATION_LOG_CSV)
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        stats["load_ok"] = int(val_df["load_ok"].sum())
        stats["ode_ok"] = int(val_df["ode_ok"].sum())
        stats["ssa_consistent"] = int(val_df["ssa_consistent"].sum())
        stats["ode_cross_ok"] = int(val_df["ode_cross_ok"].sum())
        stats["ssa_cross_ok"] = int(val_df["ssa_cross_ok"].sum())
    else:
        stats["load_ok"] = 0
        stats["ode_ok"] = 0
        stats["ssa_consistent"] = 0
        stats["ode_cross_ok"] = 0
        stats["ssa_cross_ok"] = 0

    return stats


def generate_report_md(catalog: pd.DataFrame, stats: dict) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# BioModels SSA Benchmark Report")
    lines.append("")
    lines.append("## Pipeline Funnel")
    lines.append("")
    lines.append("| Stage | Count |")
    lines.append("|-------|------:|")
    lines.append(f"| SBML downloaded | {stats['sbml_downloaded']} |")
    lines.append(f"| Passed filters (candidates) | {stats['candidates']} |")
    lines.append(f"| SBML → BNGL (atomizer) | {stats['atomized']} |")
    lines.append(f"| BNGL → .net (BNG2.pl) | {stats['net_generated']} |")
    lines.append(f"| Loads in BNGsim | {stats['load_ok']} |")
    lines.append(f"| ODE runs clean | {stats['ode_ok']} |")
    lines.append(f"| SSA self-consistent | {stats['ssa_consistent']} |")
    lines.append(f"| ODE cross-validated (vs RR) | {stats['ode_cross_ok']} |")
    if stats["ssa_cross_ok"] > 0:
        lines.append(f"| SSA ensemble cross-validated | {stats['ssa_cross_ok']} |")
    lines.append("")

    # Validated models summary
    valid = catalog[catalog["load_ok"] & catalog["ode_ok"] & catalog["ssa_consistent"]]
    n_valid = len(valid)

    lines.append(f"## Benchmark Models: {n_valid}")
    lines.append("")

    if n_valid > 0 and "net_n_species" in valid.columns:
        lines.append("### Size Distribution")
        lines.append("")
        lines.append(
            f"- Species: "
            f"median={valid['net_n_species'].median():.0f}, "
            f"range=[{valid['net_n_species'].min()}, "
            f"{valid['net_n_species'].max()}]"
        )
        lines.append(
            f"- Reactions: "
            f"median={valid['net_n_reactions'].median():.0f}, "
            f"range=[{valid['net_n_reactions'].min()}, "
            f"{valid['net_n_reactions'].max()}]"
        )
        lines.append("")

    if n_valid > 0:
        lines.append("### Model List")
        lines.append("")
        lines.append("| Model ID | Species | Reactions | ODE Cross | SSA Cross |")
        lines.append("|----------|--------:|----------:|----------:|----------:|")
        for _, row in valid.iterrows():
            n_sp = row.get("net_n_species", "?")
            n_rx = row.get("net_n_reactions", "?")
            ode_x = "✓" if row.get("ode_cross_ok", False) else "—"
            ssa_x = "✓" if row.get("ssa_cross_ok", False) else "—"
            lines.append(f"| {row['model_id']} | {n_sp} | {n_rx} | {ode_x} | {ssa_x} |")
        lines.append("")

    return "\n".join(lines)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.parse_args()

    utils.setup_logging(config.LOG_LEVEL, config.LOG_FILE)

    logger.info("=" * 60)
    logger.info("Benchmark Report Generation")
    logger.info("=" * 60)

    # Build catalog
    catalog = build_benchmark_catalog()

    if catalog.empty:
        logger.error("No data to report. Run the pipeline first.")
        return

    # Save catalog
    catalog_path = Path(config.BENCHMARK_MODELS_CSV)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_path, index=False)
    logger.info(f"Saved catalog to {catalog_path}")

    # Generate funnel stats
    stats = generate_funnel_stats()

    # Generate markdown report
    report = generate_report_md(catalog, stats)
    report_path = Path(config.REPORT_MD)
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved report to {report_path}")

    # Print summary
    print()
    print(report)

    logger.info("Report generation complete!")


if __name__ == "__main__":
    main()
