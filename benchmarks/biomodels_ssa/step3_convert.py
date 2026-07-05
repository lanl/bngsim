#!/usr/bin/env python3
"""Convert candidate SBML models to BNGL → .net format.

Two-stage conversion pipeline:
  Stage 1: SBML → BNGL via atomizer (PyBioNetGen, requires Python 3.11)
  Stage 2: BNGL → .net via BNG2.pl (BioNetGen generate_network)

The atomizer is run in a separate Python 3.11 venv because it uses
the deprecated `imp` module (removed in Python 3.12) and requires
setuptools<70 for `pkg_resources`.

Output: SBML + BNGL + .net triplets in data/{sbml_candidates,bngl_models,net_models}/

Usage:
    python step3_convert.py
    python step3_convert.py --retry-failures
    python step3_convert.py --only-atomize   # Stage 1 only
    python step3_convert.py --only-bng       # Stage 2 only (if BNGL exists)
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import config
import pandas as pd
import utils
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─── Python 3.11 venv for atomizer ──────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
VENV311 = SCRIPT_DIR / ".venv311"
ATOMIZER_CMD = VENV311 / "bin" / "bionetgen"


def check_atomizer_available() -> bool:
    """Check that the Python 3.11 venv with atomizer exists."""
    if not ATOMIZER_CMD.exists():
        logger.error(
            f"Atomizer not found at {ATOMIZER_CMD}.\n"
            f"Create the Python 3.11 venv with:\n"
            f"  uv venv --python 3.11 {VENV311}\n"
            f"  uv pip install --python {VENV311}/bin/python3.11 "
            f"bionetgen 'setuptools<70' 'python-libsbml==5.20.4'"
        )
        return False
    return True


def check_bng2pl_available() -> bool:
    """Check that BNG2.pl exists."""
    bng2pl = Path(config.BNG2_PL)
    if not bng2pl.exists():
        logger.error(f"BNG2.pl not found at {bng2pl}.\nSet BNGPATH environment variable.")
        return False
    return True


# ─── Stage 1: SBML → BNGL (atomizer) ───────────────────────────────────────


def atomize_sbml(
    model_id: str,
    sbml_path: Path,
    bngl_dir: Path,
    timeout_sec: int = 120,
) -> tuple[bool, str | None]:
    """Convert SBML to BNGL using PyBioNetGen atomizer.

    Runs in the Python 3.11 venv subprocess.

    Returns:
        Tuple of (success, error_msg).
    """
    bngl_dir.mkdir(parents=True, exist_ok=True)
    output_bngl = bngl_dir / f"{model_id}.bngl"

    if output_bngl.exists():
        return True, None

    try:
        result = subprocess.run(
            [
                str(ATOMIZER_CMD),
                "atomize",
                "-i",
                str(sbml_path.resolve()),
                "-o",
                str(output_bngl.resolve()),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(bngl_dir.resolve()),
        )

        if output_bngl.exists() and output_bngl.stat().st_size > 0:
            return True, None

        # Atomizer failed — extract error
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        err_lines = stderr.split("\n") if stderr else stdout.split("\n")
        err_msg = err_lines[-1][:200] if err_lines else "Unknown error"
        return False, err_msg

    except subprocess.TimeoutExpired:
        # Clean up partial output
        if output_bngl.exists():
            output_bngl.unlink()
        return False, f"Timeout ({timeout_sec}s)"

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ─── Stage 2: BNGL → .net (BNG2.pl) ────────────────────────────────────────


def ensure_generate_network(bngl_path: Path) -> Path:
    """Ensure the BNGL file has a generate_network action.

    The atomizer produces BNGL without actions. We need to
    append 'generate_network({overwrite=>1})' after 'end model'.

    Returns path to the (possibly modified) BNGL file.
    """
    content = bngl_path.read_text()

    # Check if generate_network already present
    if "generate_network" in content:
        return bngl_path

    # The atomizer's BNGL ends with 'end model' (no trailing newline).
    # Append the action after end model.
    if "end model" in content:
        content = content.rstrip() + "\n\ngenerate_network({overwrite=>1})\n"
    else:
        # No 'end model' — append at end
        content = content.rstrip() + "\n\ngenerate_network({overwrite=>1})\n"

    bngl_path.write_text(content)
    return bngl_path


def run_bng2pl(
    model_id: str,
    bngl_path: Path,
    net_dir: Path,
    timeout_sec: int = 60,
) -> tuple[bool, str | None]:
    """Run BNG2.pl to generate .net from BNGL.

    BNG2.pl writes the .net file in the same directory as the .bngl,
    so we run it in a temp location and copy the .net to net_dir.

    Returns:
        Tuple of (success, error_msg).
    """
    net_dir.mkdir(parents=True, exist_ok=True)
    output_net = net_dir / f"{model_id}.net"

    if output_net.exists():
        return True, None

    try:
        # BNG2.pl writes .net next to .bngl
        expected_net = bngl_path.with_suffix(".net")

        result = subprocess.run(
            [
                "perl",
                str(config.BNG2_PL),
                bngl_path.name,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(bngl_path.parent.resolve()),
        )

        if expected_net.exists() and expected_net.stat().st_size > 0:
            # Copy .net to output directory
            shutil.copy2(expected_net, output_net)
            return True, None

        # BNG2.pl failed
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        # Look for error lines in stdout (BNG2 often puts errors there)
        error_lines = []
        for line in (stdout + "\n" + stderr).split("\n"):
            ll = line.lower()
            if "error" in ll or "failed" in ll or "cannot" in ll:
                error_lines.append(line.strip())
        err_msg = "; ".join(error_lines[:3]) if error_lines else "No .net produced"
        return False, err_msg[:200]

    except subprocess.TimeoutExpired:
        return False, f"Timeout ({timeout_sec}s)"

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ─── Combined pipeline ──────────────────────────────────────────────────────


def count_net_species(net_path: str) -> int:
    """Count species in a .net file."""
    try:
        with open(net_path) as f:
            content = f.read()
        in_section = False
        count = 0
        for line in content.split("\n"):
            if "begin species" in line:
                in_section = True
                continue
            if "end species" in line:
                break
            if in_section and line.strip() and not line.strip().startswith("#"):
                count += 1
        return count
    except Exception:
        return 0


def count_net_reactions(net_path: str) -> int:
    """Count reactions in a .net file."""
    try:
        with open(net_path) as f:
            content = f.read()
        in_section = False
        count = 0
        for line in content.split("\n"):
            if "begin reactions" in line:
                in_section = True
                continue
            if "end reactions" in line:
                break
            if in_section and line.strip() and not line.strip().startswith("#"):
                count += 1
        return count
    except Exception:
        return 0


def convert_all_models(
    retry_failures: bool = False,
    only_atomize: bool = False,
    only_bng: bool = False,
) -> pd.DataFrame:
    """Convert all candidate SBML models to BNGL → .net.

    Returns DataFrame with conversion results.
    """
    candidates_dir = Path(config.SBML_CANDIDATES_DIR)
    if not candidates_dir.exists():
        logger.error(f"Candidates directory not found: {candidates_dir}")
        return pd.DataFrame()

    sbml_files = sorted(candidates_dir.glob("*.xml"))
    model_ids = [f.stem for f in sbml_files]

    if not model_ids:
        logger.error("No candidate models found")
        return pd.DataFrame()

    bngl_dir = Path(config.BNGL_MODELS_DIR)
    net_dir = Path(config.NET_MODELS_DIR)

    # Check existing conversion log for skip logic
    log_path = Path(config.CONVERSION_LOG_CSV)
    existing_log = None
    skip_ids = set()

    if log_path.exists() and not retry_failures:
        existing_log = pd.read_csv(log_path)
        # Skip models that already have .net success
        skip_ids = set(existing_log[existing_log["net_success"]]["model_id"])
        logger.info(f"Skipping {len(skip_ids)} already converted")

    todo_ids = [m for m in model_ids if m not in skip_ids]

    if not todo_ids:
        logger.info("All models already converted!")
        return existing_log if existing_log is not None else pd.DataFrame()

    logger.info(f"Converting {len(todo_ids)} models...")

    results = []
    for model_id in tqdm(todo_ids, desc="Converting"):
        sbml_path = candidates_dir / f"{model_id}.xml"

        row = {
            "model_id": model_id,
            "atomize_success": False,
            "atomize_error": None,
            "net_success": False,
            "bng_error": None,
            "convert_error": None,
            "net_n_species": 0,
            "net_n_reactions": 0,
        }

        # ── Stage 1: SBML → BNGL ──
        if not only_bng:
            ok, err = atomize_sbml(
                model_id,
                sbml_path,
                bngl_dir,
                timeout_sec=config.ATOMIZE_TIMEOUT,
            )
            row["atomize_success"] = ok
            row["atomize_error"] = err

            if not ok:
                row["convert_error"] = f"Atomizer: {err}"
                results.append(row)
                continue

        # Check if BNGL exists (for --only-bng mode)
        bngl_path = bngl_dir / f"{model_id}.bngl"
        if not bngl_path.exists():
            if only_bng:
                row["convert_error"] = "No BNGL file (run without --only-bng first)"
            else:
                row["convert_error"] = "BNGL file missing after atomizer"
            results.append(row)
            continue

        row["atomize_success"] = True

        if only_atomize:
            results.append(row)
            continue

        # ── Prepare BNGL for generate_network ──
        ensure_generate_network(bngl_path)

        # ── Stage 2: BNGL → .net ──
        ok, err = run_bng2pl(
            model_id,
            bngl_path,
            net_dir,
            timeout_sec=config.GENERATE_NET_TIMEOUT,
        )
        row["net_success"] = ok
        row["bng_error"] = err

        if not ok:
            row["convert_error"] = f"BNG2.pl: {err}"
        else:
            net_path = str(net_dir / f"{model_id}.net")
            row["net_n_species"] = count_net_species(net_path)
            row["net_n_reactions"] = count_net_reactions(net_path)

        results.append(row)

    df = pd.DataFrame(results)

    # Merge with existing log
    if existing_log is not None and not df.empty:
        # Align columns
        for col in existing_log.columns:
            if col not in df.columns:
                df[col] = None
        for col in df.columns:
            if col not in existing_log.columns:
                existing_log[col] = None
        df = pd.concat([existing_log, df], ignore_index=True)
        df = df.drop_duplicates(subset=["model_id"], keep="last")

    return df


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Convert SBML → BNGL → .net")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry previously failed conversions",
    )
    parser.add_argument(
        "--only-atomize",
        action="store_true",
        help="Only run Stage 1 (SBML → BNGL)",
    )
    parser.add_argument(
        "--only-bng",
        action="store_true",
        help="Only run Stage 2 (BNGL → .net)",
    )

    args = parser.parse_args()

    utils.setup_logging(config.LOG_LEVEL, config.LOG_FILE)

    logger.info("=" * 60)
    logger.info("SBML → BNGL → .net Conversion Pipeline")
    logger.info("=" * 60)

    # Check prerequisites
    if not args.only_bng and not check_atomizer_available():
        return
    if not args.only_atomize and not check_bng2pl_available():
        return

    logger.info(f"Atomizer: {ATOMIZER_CMD}")
    logger.info(f"BNG2.pl:  {config.BNG2_PL}")

    df = convert_all_models(
        retry_failures=args.retry_failures,
        only_atomize=args.only_atomize,
        only_bng=args.only_bng,
    )

    if df.empty:
        logger.error("No models to convert. Run step2_filter.py first.")
        return

    # Save conversion log
    log_path = Path(config.CONVERSION_LOG_CSV)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(log_path, index=False)
    logger.info(f"Saved conversion log to {log_path}")

    # Print summary
    total = len(df)
    n_atomized = int(df["atomize_success"].sum())
    n_netted = int(df["net_success"].sum())

    print()
    print("Conversion Summary")
    print("=" * 50)
    print(f"  Total candidates:      {total}")
    print(f"  SBML → BNGL OK:        {n_atomized}")
    print(f"  BNGL → .net OK:        {n_netted}")
    print(f"  Full pipeline success:  {n_netted}/{total}")
    print()

    # Show atomizer failures
    atom_fail = df[~df["atomize_success"]]
    if len(atom_fail) > 0:
        print(f"Atomizer failures ({len(atom_fail)}):")
        for _, row in atom_fail.head(10).iterrows():
            err = row.get("atomize_error") or "unknown"
            print(f"  {row['model_id']}: {str(err)[:80]}")
        print()

    # Show BNG2.pl failures
    bng_fail = df[df["atomize_success"] & ~df["net_success"]]
    if len(bng_fail) > 0:
        print(f"BNG2.pl failures ({len(bng_fail)}):")
        for _, row in bng_fail.head(10).iterrows():
            err = row.get("bng_error") or "unknown"
            print(f"  {row['model_id']}: {str(err)[:80]}")
        print()

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
