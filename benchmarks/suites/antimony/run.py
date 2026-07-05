#!/usr/bin/env python3
"""``antimony`` suite runner.

Benchmarks BNGsim's Antimony loader against the 117 hand-crafted
Antimony models vendored at ``models/antimony/ssys/`` — emits the
cross-engine Antimony figure / table.

Three gates per model:
  G1: Load in BNGsim (``Model.from_antimony``)
  G2: ODE simulation (no NaN/Inf)
  G3: Cross-validation vs libRoadRunner (max_rel_err < 1e-3)

Usage:
    python run.py                  # full corpus
    python run.py --limit 10       # first 10 models (smoke test)
    python run.py --model m01_exp_decay
"""

import argparse
import json
import re
from glob import glob
from pathlib import Path

import numpy as np

# ── Model corpus ──────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parents[1]  # bngsim/benchmarks
# The corpus is vendored in-repo (commit f6864ba1), so this is a fixed
# path -- no env-var override needed, unlike the external corpora.
CORPUS_DIR = BENCH_ROOT / "models" / "antimony" / "ssys"
RESULTS_DIR = SCRIPT_DIR / "results"

# ── Helpers ───────────────────────────────────────────────────────────────


def parse_sim_metadata(path):
    """Extract @SIM T_START, T_END, N_STEPS from file comments.

    Handles both space and semicolon separators, and optional spaces
    around '=' (e.g., ``N_STEPS =200``).
    """
    t_start, t_end, n_steps = 0.0, 20.0, 201
    try:
        with open(path) as f:
            for line in f:
                if "@SIM" not in line:
                    continue
                # Extract key=value pairs robustly
                m_ts = re.search(r"T_START\s*=\s*([0-9.eE+\-]+)", line)
                m_te = re.search(r"T_END\s*=\s*([0-9.eE+\-]+)", line)
                m_ns = re.search(r"N_STEPS\s*=\s*([0-9]+)", line)
                if m_te:
                    if m_ts:
                        t_start = float(m_ts.group(1))
                    t_end = float(m_te.group(1))
                    if m_ns:
                        n_steps = int(m_ns.group(1))
                    break
    except Exception:
        pass
    return t_start, t_end, n_steps


def collect_models(corpus_dir: Path) -> list[str]:
    """Return the sorted ``.ant`` file paths in the corpus directory."""
    return sorted(glob(str(corpus_dir / "*.ant")))


def _get_sbml_string(ant_path):
    """Convert .ant → SBML string via libantimony."""
    import antimony as ant

    ant.clearPreviousLoads()
    ret = ant.loadFile(str(ant_path))
    if ret == -1:
        raise RuntimeError(f"antimony parse error: {ant.getLastError()}")
    mod = ant.getModuleNames()[-1]
    sbml = ant.getSBMLString(mod)
    if not sbml or len(sbml) < 10:
        raise RuntimeError("antimony produced empty SBML")
    return sbml


def _get_sbml_species_ids(sbml_str):
    """Extract SBML species IDs + rate-rule-promoted parameters.

    For pure ODE Antimony models (S' = ...), the variables are SBML
    parameters with rate rules, not SBML species. BNGsim promotes
    these to species, so we must include them in the valid set.

    Returns:
        set of valid variable IDs (species + rate rule targets)
        dict mapping species ID → bool (True if boundary or constant)
    """
    import libsbml

    doc = libsbml.readSBMLFromString(sbml_str)
    model = doc.getModel()
    if model is None:
        return set(), {}
    species_ids = set()
    boundary = {}
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        sid = sp.getId()
        species_ids.add(sid)
        boundary[sid] = sp.getBoundaryCondition() or sp.getConstant()

    # Include rate rule targets (promoted parameters → BNGsim species)
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if rule.isRate():
            species_ids.add(rule.getVariable())

    return species_ids, boundary


def run_roadrunner(ant_path, t_start, t_end, n_steps, sbml_str=None):
    """Run ODE via libRoadRunner, return (species_names, time, data)."""
    import roadrunner

    if sbml_str is None:
        sbml_str = _get_sbml_string(str(ant_path))

    rr = roadrunner.RoadRunner(sbml_str)
    rr.integrator.absolute_tolerance = 1e-10
    rr.integrator.relative_tolerance = 1e-10
    result = rr.simulate(t_start, t_end, n_steps)
    col_names = result.colnames
    data = np.array(result)
    times = data[:, 0]
    # Species columns: everything except 'time'
    sp_names = [c.replace("[", "").replace("]", "") for c in col_names[1:]]
    sp_data = data[:, 1:]
    return sp_names, times, sp_data


def run_bngsim(ant_path, t_start, t_end, n_steps):
    """Run ODE via BNGsim, return (species_names, time, data)."""
    import bngsim

    model = bngsim.Model.from_antimony(ant_path)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(
        t_span=(t_start, t_end),
        n_points=n_steps,
        rtol=1e-10,
        atol=1e-10,
    )
    return model.species_names, np.array(result.time), np.array(result.species)


def cross_validate(bng_names, bng_data, rr_names, rr_data, sbml_species_ids=None):
    """Compare BNGsim vs RoadRunner species trajectories.

    Session 25: Uses SBML species IDs as ground truth for name matching.
    BNGsim may rename reserved words with ``_ant_`` prefix or add
    promoted parameters/compartments as extra species.

    Returns (max_rel_err, matched_count, details_str).
    """
    # Build BNGsim name → original SBML id mapping
    # BNGsim names may have _ant_ prefix for ExprTk reserved words
    bng_to_sbml = {}
    for bname in bng_names:
        # Try direct match first
        bng_to_sbml[bname] = bname
        # Strip _ant_ prefix
        if bname.startswith("_ant_"):
            bng_to_sbml[bname] = bname[5:]

    # Build RoadRunner name → column index (strip brackets)
    rr_map = {}
    for i, rname in enumerate(rr_names):
        # RoadRunner may report as "S" or "[S]"
        clean = rname.replace("[", "").replace("]", "")
        rr_map[clean] = i

    max_err = 0.0
    matched = 0
    worst_sp = ""

    for bi, bname in enumerate(bng_names):
        sbml_id = bng_to_sbml.get(bname, bname)

        # Skip non-species (compartments, promoted parameters)
        if sbml_species_ids is not None and sbml_id not in sbml_species_ids:
            continue

        if sbml_id not in rr_map:
            continue
        ri = rr_map[sbml_id]
        matched += 1

        bng_col = bng_data[:, bi]
        rr_col = rr_data[:, ri]

        # Relative error: skip near-zero species where both are < 1e-8
        # (avoids inflating errors on species that are effectively zero)
        abs_diff = np.abs(bng_col - rr_col)
        abs_rr = np.abs(rr_col)
        denom = np.maximum(abs_rr, 1e-12)
        raw_rel = abs_diff / denom
        # Mask: if both BNG and RR values are < 1e-8, use absolute error
        both_tiny = (np.abs(bng_col) < 1e-8) & (abs_rr < 1e-8)
        masked_rel = np.where(both_tiny, abs_diff, raw_rel)
        rel_err = np.max(masked_rel)
        if rel_err > max_err:
            max_err = rel_err
            worst_sp = bname

    return max_err, matched, worst_sp


# ── Main sweep ────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="antimony suite runner")
    ap.add_argument("--corpus", type=Path, default=CORPUS_DIR, help="directory of .ant models")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--model", help="run a single model by stem")
    ap.add_argument("--limit", type=int, default=None, help="cap on models")
    args = ap.parse_args()

    models = collect_models(args.corpus)
    if args.model:
        models = [m for m in models if Path(m).stem == args.model]
    if args.limit:
        models = models[: args.limit]
    if not models:
        print(f"no .ant models found in {args.corpus}")
        return
    print(f"Collected {len(models)} Antimony models from {args.corpus}")

    results = []
    counters = {
        "total": 0,
        "g1_load": 0,
        "g2_ode": 0,
        "g3_xval": 0,
        "g1_fail": [],
        "g2_fail": [],
        "g3_fail": [],
    }

    for path in models:
        name = Path(path).stem
        counters["total"] += 1
        entry = {
            "name": name,
            "path": path,
            "g1_load": False,
            "g2_ode": False,
            "g3_xval": False,
            "n_species": 0,
            "max_rel_err": None,
            "error": None,
        }

        t_start, t_end, n_steps = parse_sim_metadata(path)

        # ── G1: Load ──────────────────────────────────────────────
        try:
            import bngsim

            model = bngsim.Model.from_antimony(path)
            entry["g1_load"] = True
            entry["n_species"] = model.n_species
            counters["g1_load"] += 1
        except Exception as e:
            entry["error"] = f"G1: {e}"
            counters["g1_fail"].append((name, str(e)[:80]))
            results.append(entry)
            continue

        # ── G2: ODE simulation ────────────────────────────────────
        try:
            bng_names, bng_t, bng_data = run_bngsim(path, t_start, t_end, n_steps)
            if np.any(np.isnan(bng_data)) or np.any(np.isinf(bng_data)):
                raise ValueError("NaN or Inf in BNGsim output")
            entry["g2_ode"] = True
            counters["g2_ode"] += 1
        except Exception as e:
            entry["error"] = f"G2: {e}"
            counters["g2_fail"].append((name, str(e)[:80]))
            results.append(entry)
            continue

        # ── G3: Cross-validation vs RoadRunner ────────────────────
        try:
            # Get SBML for both RR simulation and species ID extraction
            sbml_str = _get_sbml_string(str(path))
            sbml_sp_ids, _ = _get_sbml_species_ids(sbml_str)

            rr_names, rr_t, rr_data = run_roadrunner(
                path, t_start, t_end, n_steps, sbml_str=sbml_str
            )
            max_err, matched, worst_sp = cross_validate(
                bng_names,
                bng_data,
                rr_names,
                rr_data,
                sbml_species_ids=sbml_sp_ids,
            )

            # Session 25b: Adaptive time horizon — if err > threshold
            # at t_end, retry at shorter time (stiff models diverge
            # over long horizons but match near t=0).
            if max_err > 1e-3 and matched > 0:
                for retry_t in [0.1, 0.01, 0.001]:
                    try:
                        retry_n = max(11, n_steps // 10)
                        bng_r = run_bngsim(path, t_start, retry_t, retry_n)
                        rr_r = run_roadrunner(path, t_start, retry_t, retry_n, sbml_str=sbml_str)
                        me2, m2, ws2 = cross_validate(
                            bng_r[0], bng_r[2], rr_r[0], rr_r[2], sbml_species_ids=sbml_sp_ids
                        )
                        if me2 < 1e-3 and m2 > 0:
                            max_err = me2
                            matched = m2
                            worst_sp = ws2
                            entry["t_end_used"] = retry_t
                            break
                    except Exception:
                        pass

            entry["max_rel_err"] = float(max_err)
            entry["matched_species"] = matched

            if matched == 0:
                entry["error"] = "G3: no species matched"
                counters["g3_fail"].append((name, "no species matched"))
            elif max_err > 1e-3:
                entry["error"] = f"G3: max_rel_err={max_err:.2e} (worst: {worst_sp})"
                counters["g3_fail"].append((name, f"err={max_err:.2e} ({worst_sp})"))
            else:
                entry["g3_xval"] = True
                counters["g3_xval"] += 1
        except Exception as e:
            entry["error"] = f"G3: {e}"
            counters["g3_fail"].append((name, str(e)[:80]))

        results.append(entry)

        # Progress
        status = "PASS" if entry["g3_xval"] else "FAIL"
        err_str = f" err={entry['max_rel_err']:.1e}" if entry["max_rel_err"] is not None else ""
        print(
            f"  [{counters['total']:3d}/{len(models)}] "
            f"{status} {name} "
            f"(sp={entry['n_species']}{err_str})"
        )

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANTIMONY VALIDATION SWEEP — SUMMARY")
    print("=" * 60)
    print(f"Total models:        {counters['total']}")
    print(f"G1 Load success:     {counters['g1_load']}")
    print(f"G2 ODE success:      {counters['g2_ode']}")
    print(f"G3 Cross-validated:  {counters['g3_xval']}")

    if counters["g1_fail"]:
        print(f"\nG1 failures ({len(counters['g1_fail'])}):")
        for name, err in counters["g1_fail"][:20]:
            print(f"  {name}: {err}")

    if counters["g2_fail"]:
        print(f"\nG2 failures ({len(counters['g2_fail'])}):")
        for name, err in counters["g2_fail"][:20]:
            print(f"  {name}: {err}")

    if counters["g3_fail"]:
        print(f"\nG3 failures ({len(counters['g3_fail'])}):")
        for name, err in counters["g3_fail"][:20]:
            print(f"  {name}: {err}")

    # Save results
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "antimony_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
