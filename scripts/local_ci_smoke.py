"""Functional smoke checks for an installed bngsim wheel.

Designed to be invoked from a freshly installed wheel — no pytest, no editable
source. Exercises every loader, every backend session, and capabilities()
introspection. Writes a JSON report so the parent orchestrator can render a
table.

Usage:
    python local_ci_smoke.py --data-dir <bngsim/tests/data>
                             --antimony-fixture-dir <bngsim/benchmarks/models/antimony/ssys>
                             --report <path/to/smoke.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def resolve_antimony_fixture(base_dir: Path) -> Path:
    """Accept the current corpus dir, or a broader parent path, and find the smoke fixture."""
    candidates = (
        base_dir / "m01_exp_decay.ant",
        base_dir / "ssys" / "m01_exp_decay.ant",
        base_dir / "models" / "antimony" / "ssys" / "m01_exp_decay.ant",
        base_dir / "benchmarks" / "models" / "antimony" / "ssys" / "m01_exp_decay.ant",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--antimony-fixture-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument(
        "--require-klu",
        action="store_true",
        help="Fail if the sparse KLU solver is not compiled in (GH #209: Linux "
        "wheels must ship it, or large models silently fall back to dense O(N^3)).",
    )
    args = ap.parse_args()

    results: dict[str, dict] = {}

    def record(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        results[name] = {"ok": ok, "detail": detail}
        print(f"  [{status}] {name} {detail}".rstrip())

    print(f"Python: {sys.version.split()[0]}")
    try:
        import bngsim

        print(f"bngsim: {bngsim.__version__}")
    except Exception:
        traceback.print_exc()
        args.report.write_text(
            json.dumps({"import": {"ok": False, "detail": "import bngsim failed"}})
        )
        return 1

    print("=== capabilities ===")
    try:
        caps = bngsim.capabilities()
        feats = caps.get("features", {})
        print(f"  features: {feats}")
        print(f"  missing: {caps.get('missing', {})}")
        record("capabilities", True, f"features={feats}")
    except Exception as e:
        record("capabilities", False, repr(e))

    if args.require_klu:
        # GH #209: the Linux wheel must ship the sparse KLU solver. Asserting it
        # here — against the manylinux ABI the published wheel targets — is where
        # a mis-bundled SuiteSparse dylib would actually fail to dlopen.
        has_klu = bool(getattr(bngsim, "HAS_KLU", False))
        record(
            "KLU (required)",
            has_klu,
            "sparse solver compiled in" if has_klu else "MISSING (wheel is dense-only)",
        )

    print("=== .net ODE ===")
    try:
        m = bngsim.Model.from_net(str(args.data_dir / "simple_decay.net"))
        sim = bngsim.Simulator(m, method="ode")
        r = sim.run(t_span=(0, 10), n_points=11)
        ok = r.n_times == 11 and r.n_observables >= 1
        record(".net+ODE", ok, f"shape=({r.n_times},{r.n_observables})")
    except Exception as e:
        record(".net+ODE", False, repr(e))

    print("=== SBML ODE ===")
    try:
        m = bngsim.Model.from_sbml(str(args.data_dir / "BIOMD0000000003.xml"))
        sim = bngsim.Simulator(m, method="ode")
        r = sim.run(t_span=(0, 1), n_points=11)
        ok = r.n_times == 11
        record("SBML+ODE", ok, f"shape=({r.n_times},{r.n_observables})")
    except Exception as e:
        record("SBML+ODE", False, repr(e))

    print("=== NFsim session ===")
    HAS_NFSIM = getattr(bngsim, "HAS_NFSIM", False)
    if HAS_NFSIM:
        try:
            xml = args.data_dir / "nfsim" / "simple_system.xml"
            with bngsim.NfsimSession(str(xml)) as nf:
                nf.initialize(seed=42)
                nf.simulate(0, 1, n_points=11)
            record("NFsim session", True, "")
        except Exception as e:
            record("NFsim session", False, repr(e))
    else:
        record("NFsim session", False, "HAS_NFSIM=False (compiled without)")

    print("=== RuleMonkey session ===")
    HAS_RULEMONKEY = getattr(bngsim, "HAS_RULEMONKEY", False)
    if HAS_RULEMONKEY:
        try:
            xml = args.data_dir / "nfsim" / "simple_system.xml"
            with bngsim.RuleMonkeySession(str(xml)) as rm:
                rm.initialize(seed=42)
                r = rm.simulate(0, 1, n_points=11)
            ok = r.n_times == 11
            record("RuleMonkey session", ok, "")
        except Exception as e:
            record("RuleMonkey session", False, repr(e))
    else:
        record("RuleMonkey session", False, "HAS_RULEMONKEY=False (compiled without)")

    print("=== Antimony (optional) ===")
    has_antimony = getattr(bngsim, "HAS_ANTIMONY", False)
    if has_antimony:
        ant = resolve_antimony_fixture(args.antimony_fixture_dir)
        if not ant.exists():
            record("Antimony", False, f"fixture missing: {ant}")
        else:
            try:
                m = bngsim.Model.from_antimony(str(ant))
                sim = bngsim.Simulator(m, method="ode")
                r = sim.run(t_span=(0, 1), n_points=11)
                record("Antimony", True, f"shape=({r.n_times},{r.n_observables})")
            except Exception as e:
                record("Antimony", False, repr(e))
    else:
        # Not a hard failure: antimony is an optional extra and may legitimately
        # be unavailable on a given platform/Python (no wheel published).
        record("Antimony", True, "HAS_ANTIMONY=False (extra not installed)")

    args.report.write_text(json.dumps(results, indent=2) + "\n")
    fail = [k for k, v in results.items() if not v["ok"]]
    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k:<22} {'PASS' if v['ok'] else 'FAIL'} {v.get('detail', '')}".rstrip())
    if fail:
        print(f"\nFAILURES: {fail}")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
