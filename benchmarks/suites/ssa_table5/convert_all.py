#!/usr/bin/env python3
"""Convert every ssa_table5 model with bngsim's OWN converter (never BNG2.pl) and
cache the cross-engine feed artifacts, logging each ConversionReport.

  * 8 BNGL  .net -> SBML   (results/converted/<name>.xml)   feeds RoadRunner + COPASI
  * 6 SBML  .xml -> .net   (results/converted/<id>.net)     feeds run_network

Coverage is fixed HERE (per the brief): a model that does not convert faithfully
for an engine gets that engine's cell marked N/A + reason, not forced.

Faithfulness gates (mirroring parity_checks/bng_parity/net_roadrunner.py):
  * RoadRunner  needs: strict convert OK, max_rhs_delta<=RHS_TOL, NO repeated
    reactant (2A->C: SBML law k*A*A != exact propensity k*A*(A-1); RR-gillespie
    fires the wrong propensity, GH #9), and NO dropped time-triggered events
    (RR-gillespie silently won't fire them).
  * COPASI      needs: strict convert OK, max_rhs_delta<=RHS_TOL. COPASI's exact
    stochastic solver derives correct combinatorial propensities for higher-order
    reactions and DOES fire events, so neither the repeated-reactant nor the event
    gate applies to it.
  * run_network needs (SBML->.net direction): convert OK and NO dropped events
    (a dropped event changes the .net's dynamics vs the source SBML).

Writes results/converted/conversion_log.json, then checks the verdicts it just
computed against the coverage table _ssa_config.py declares -- that table is what
the orchestrator obeys, and it is hand-written, so a conversion that starts or
stops being faithful (a re-pointed model, a converter fix) has to show up here
rather than as a silently unfaithful cell in the timing matrix. Exits non-zero on
a disagreement, naming the cell.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _ssa_config as C  # noqa: E402

CORPUS = json.loads((HERE / "corpus.json").read_text())
OUT = HERE / "results" / "converted"
RHS_TOL = 1e-6  # net_roadrunner.RHS_FAITHFUL_TOL


def _has_repeated_reactant(sbml_text: str) -> bool:
    """True iff some reaction consumes a species with total multiplicity >=2
    (homo-oligomerization). Copied from net_roadrunner._has_repeated_reactant."""
    import libsbml

    doc = libsbml.readSBMLFromString(sbml_text)
    model = doc.getModel()
    if model is None:
        return False
    for i in range(model.getNumReactions()):
        r = model.getReaction(i)
        tally: dict[str, float] = {}
        for j in range(r.getNumReactants()):
            sr = r.getReactant(j)
            tally[sr.getSpecies()] = tally.get(sr.getSpecies(), 0.0) + sr.getStoichiometry()
        if any(v >= 1.5 for v in tally.values()):
            return True
    return False


def _report_dict(rep):
    st = rep.structural
    return {
        "n_species": rep.n_species,
        "n_reactions": rep.n_reactions,
        "n_parameters": rep.n_parameters,
        "n_observables": rep.n_observables,
        "max_rhs_delta": rep.max_rhs_delta,
        "rhs_faithful": rep.rhs_faithful,
        "structural_passed": bool(getattr(st, "passed", False)),
        "lossy": list(rep.lossy or []),
        "dropped": list(rep.dropped or []),
    }


def _delta_ok(rep) -> bool:
    d = rep.max_rhs_delta
    return d is not None and d == d and abs(d) <= RHS_TOL  # d==d guards NaN


def _dropped_events(rep) -> bool:
    return any("event" in str(x).lower() for x in (rep.dropped or []))


def convert_bngl(m) -> dict:
    """net -> SBML for one BNGL model; returns a coverage/report record."""
    from bngsim.convert import net_to_sbml

    name = m["name"]
    net = HERE / m["file"]
    xml = OUT / f"{name}.xml"
    rec = {
        "name": name,
        "direction": "net_to_sbml",
        "src": m["file"],
        "out": str(xml.relative_to(HERE)),
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = net_to_sbml(net, xml, validate="L1", strict=True)
    except Exception as e:  # noqa: BLE001
        rec.update(
            ok=False,
            error=f"{type(e).__name__}: {e}"[:300],
            rr="N/A (net->SBML strict-convert failed)",
            copasi="N/A (net->SBML strict-convert failed)",
        )
        return rec
    rec.update(ok=True, error="", **_report_dict(rep))
    repeated = _has_repeated_reactant(rep.output_text)
    events = _dropped_events(rep)
    rec["repeated_reactant"] = repeated
    delta_ok = _delta_ok(rep)
    # RoadRunner gate
    if not (rep.structural.passed and delta_ok):
        rec["rr"] = f"N/A (unfaithful net->SBML: max_rhs_delta={rep.max_rhs_delta})"
    elif repeated:
        rec["rr"] = (
            "N/A (repeated reactant: SBML k*A*A != exact k*A*(A-1); RR-gillespie fires wrong propensity)"
        )
    elif events:
        rec["rr"] = "N/A (time-triggered events: RR-gillespie won't fire them)"
    else:
        rec["rr"] = "ok"
    # COPASI gate
    rec["copasi"] = (
        "ok"
        if (rep.structural.passed and delta_ok)
        else f"N/A (unfaithful net->SBML: max_rhs_delta={rep.max_rhs_delta})"
    )
    return rec


def convert_sbml(m) -> dict:
    """SBML -> .net for one SBML model; returns a coverage/report record for run_network."""
    from bngsim.convert import sbml_to_net

    mid = m["id"]
    xml = HERE / m["file"]
    net = OUT / f"{mid}.net"
    rec = {
        "name": mid,
        "direction": "sbml_to_net",
        "src": m["file"],
        "out": str(net.relative_to(HERE)),
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = sbml_to_net(xml, net, validate="L2", strict=True)
    except Exception as e:  # noqa: BLE001
        rec.update(
            ok=False,
            error=f"{type(e).__name__}: {e}"[:300],
            run_network="N/A (SBML->.net strict-convert failed)",
        )
        return rec
    rec.update(ok=True, error="", **_report_dict(rep))
    events = _dropped_events(rep)
    if not rep.structural.passed:
        rec["run_network"] = "N/A (SBML->.net structural mismatch)"
    elif events:
        rec["run_network"] = (
            "N/A (SBML->.net dropped time-triggered event(s); .net dynamics differ from source SBML)"
        )
    else:
        rec["run_network"] = "ok"
    return rec


def check_coverage(log) -> list[str]:
    """Compare the verdicts just computed against _ssa_config.COVERAGE."""
    problems = []
    for direction, engines in (("bngl", ("rr", "copasi")), ("sbml", ("run_network",))):
        for rec in log[direction]:
            for short in engines:
                engine = {"rr": "roadrunner"}.get(short, short)
                computed_ok = rec.get(short) == "ok"
                declared_ok = C.cell_status(rec["name"], engine)[0] == "ok"
                if computed_ok != declared_ok:
                    verb = (
                        "is faithful but is declared"
                        if computed_ok
                        else "is not faithful but is declared"
                    )
                    problems.append(
                        f"{rec['name']}/{engine}: conversion {verb} "
                        f"{C.cell_status(rec['name'], engine)[0]!r} in _ssa_config.COVERAGE "
                        f"(convert_all says: {rec.get(short)})"
                    )
    return problems


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import bngsim

    log = {"bngsim_version": bngsim.__version__, "rhs_tol": RHS_TOL, "bngl": [], "sbml": []}
    print(f"convert_all  bngsim {bngsim.__version__}\n")
    print("== BNGL -> SBML (feeds RoadRunner + COPASI) ==")
    for m in CORPUS["bngl"]:
        rec = convert_bngl(m)
        log["bngl"].append(rec)
        print(
            f"  {rec['name']:24} ok={rec.get('ok')} delta={rec.get('max_rhs_delta')} "
            f"repeat={rec.get('repeated_reactant')} rr={rec['rr'][:30]} copasi={rec['copasi'][:20]}"
            + (f"  ERR {rec['error']}" if not rec.get("ok") else "")
        )
    print("\n== SBML -> .net (feeds run_network) ==")
    for m in CORPUS["sbml"]:
        rec = convert_sbml(m)
        log["sbml"].append(rec)
        print(
            f"  {rec['name']:20} ok={rec.get('ok')} delta={rec.get('max_rhs_delta')} "
            f"dropped={rec.get('dropped')} run_network={rec['run_network'][:40]}"
            + (f"  ERR {rec['error']}" if not rec.get("ok") else "")
        )
    (OUT / "conversion_log.json").write_text(json.dumps(log, indent=2))
    print(f"\n-> {OUT / 'conversion_log.json'}")

    problems = check_coverage(log)
    if problems:
        print("\n== COVERAGE DISAGREEMENT (_ssa_config.COVERAGE is what the runner obeys) ==")
        for p in problems:
            print(f"  {p}")
        return 1
    print("coverage: _ssa_config.COVERAGE agrees with every conversion verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
