"""bngsim.convert._cli — console entry points for the format converters.

* :func:`main` — ``bngsim-sbml2net`` (SBML → ``.net``), also ``python -m
  bngsim.convert``.
* :func:`net2sbml_main` — ``bngsim-net2sbml`` (``.net`` → SBML), the reverse.
* :func:`sbml2bngl_main` — ``bngsim-sbml2bngl`` (SBML → compartmental ``.bngl``),
  the cBNGL writer that recovers static compartment volumes (GH #224).
* :func:`validate_main` — ``bngsim-validate-conversion`` (L0–L4 validation, GH
  #217), gating either direction.
* :func:`omex_main` — ``bngsim-omex`` (``pack`` / ``unpack``), COMBINE archive
  packaging (GH #219).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path


def _add_gate_args(
    p: argparse.ArgumentParser,
    *,
    default: str = "L1",
    choices: tuple[str, ...] = ("none", "L1", "full"),
) -> None:
    """Add the shared ``--gate``/``--t-end``/``--n-points`` validation flags.

    ``--gate`` selects how hard the conversion is checked before it is accepted:
    ``none`` skips validation, ``L1`` runs the fast structural round-trip, ``L2``
    (sbml2net only) adds a direct ODE-RHS identity self-check that refuses a
    network whose forcing the flat ``.net`` cannot carry (GH #223), and ``full``
    runs the complete L0–L4 ladder (GH #217) gating on the hard levels L0–L3 —
    "convert *and prove faithful*". ``--t-end`` / ``--n-points`` set the L3
    simulation grid used by ``--gate full``. ``default``/``choices`` are
    per-direction (sbml2net defaults to ``L2``; net2sbml's ``L1`` already does the
    RHS round-trip via :func:`validate_roundtrip`).
    """
    p.add_argument(
        "--gate",
        choices=choices,
        default=default,
        help=(
            "validation gate before accepting the conversion: none (skip), "
            f"L1 (fast structural round-trip), {'L2 (structural + ODE-RHS identity), ' if 'L2' in choices else ''}"
            "or full (the L0–L4 ladder gating on L0–L3 — proves the conversion "
            f"faithful; exits non-zero if a hard gate fails). Default: {default}"
        ),
    )
    # Back-compat alias for the pre-#217 flag: equivalent to --gate none.
    p.add_argument("--no-validate", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--t-end",
        type=float,
        default=100.0,
        help="end time for the L3 simulation comparison under --gate full (default: 100)",
    )
    p.add_argument(
        "--n-points",
        type=int,
        default=101,
        help="number of time points for the L3 comparison under --gate full (default: 101)",
    )


def _resolve_gate(args: argparse.Namespace) -> str | None:
    """Map the parsed ``--gate``/``--no-validate`` flags to a ``validate=`` value."""
    gate = "none" if args.no_validate else args.gate
    return {"none": None, "L1": "L1", "L2": "L2", "full": "full"}[gate]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bngsim-sbml2net",
        description=(
            "Convert an SBML model to a BioNetGen .net network "
            "(network channel only — species, reactions, parameters, "
            "observables, functions). Events/protocol are out of scope."
        ),
    )
    p.add_argument("sbml", type=Path, help="source SBML .xml file")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output .net path (default: alongside the input with a .net suffix)",
    )
    p.add_argument(
        "--allow-lossy",
        action="store_true",
        help=(
            "emit a best-effort network even when the model uses constructs the "
            ".net text format cannot represent faithfully (downgrades the error "
            "to a warning)"
        ),
    )
    _add_gate_args(p, default="L2", choices=("none", "L1", "L2", "full"))
    p.add_argument(
        "--sedml",
        type=Path,
        default=None,
        help=(
            "the SED-ML protocol sidecar accompanying the SBML. Its time course "
            "is parsed so --gate full's L3 check runs over the model's own horizon "
            "(avoids the blanket-grid stiff-hang) — the mirror of net2sbml's "
            "--bngl. Auto-detected as a sibling <stem>.sedml when present and not "
            "given."
        ),
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the per-conversion summary (still prints errors)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Import here so ``--help`` stays fast and import errors surface as a clean message.
    from bngsim._exceptions import ConversionError
    from bngsim.convert import sbml_to_net

    if not args.sbml.exists():
        print(f"error: no such file: {args.sbml}", file=sys.stderr)
        return 2

    out = args.out if args.out is not None else args.sbml.with_suffix(".net")

    # Use the SED-ML sidecar for the L3 horizon: explicit --sedml, else a sibling.
    sedml = args.sedml
    if sedml is None:
        sibling = args.sbml.with_suffix(".sedml")
        if sibling.exists():
            sedml = sibling
    elif not sedml.exists():
        print(f"error: no such SED-ML sidecar: {sedml}", file=sys.stderr)
        return 2

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = sbml_to_net(
                args.sbml,
                out,
                validate=_resolve_gate(args),
                strict=not args.allow_lossy,
                sedml=sedml,
                t_span=(0.0, args.t_end),
                n_points=args.n_points,
            )
    except ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    for w in caught:
        print(f"warning: {w.message}", file=sys.stderr)

    if not args.quiet:
        print(report.summary())

    # Non-zero exit if an L1 check ran and failed, so scripts/CI can gate on it.
    return 0 if report.ok else 1


def _build_net2sbml_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bngsim-net2sbml",
        description=(
            "Convert a BioNetGen .net network to SBML (Level 3 Version 2; "
            "network channel only — species, reactions, parameters, "
            "observables, functions, compartments). The reverse of sbml2net."
        ),
    )
    p.add_argument("net", type=Path, help="source BioNetGen .net file")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output SBML .xml path (default: alongside the input with a .xml suffix)",
    )
    p.add_argument(
        "--allow-lossy",
        action="store_true",
        help=(
            "emit a best-effort document even when the model uses constructs SBML "
            "cannot represent faithfully here (live/time-varying volumes, tfun "
            "table functions); downgrades the error to a warning"
        ),
    )
    _add_gate_args(p)
    p.add_argument(
        "--bngl",
        type=Path,
        default=None,
        help=(
            "the source .bngl the .net was generated from. Its simulate protocol "
            "is parsed so --gate full's L3 check runs over the model's own horizon "
            "(avoids the blanket-grid stiff-hang). Auto-detected as a sibling "
            "<stem>.bngl when present and not given."
        ),
    )
    p.add_argument(
        "--sidecar",
        action="store_true",
        help=(
            "also emit a SED-ML simulation-protocol sidecar (GH #218) next to the "
            "SBML (default <stem>.sedml): a uniform time course reporting every "
            "observable, since SBML carries no protocol of its own"
        ),
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the per-conversion summary (still prints errors)",
    )
    return p


def net2sbml_main(argv: list[str] | None = None) -> int:
    args = _build_net2sbml_parser().parse_args(argv)
    from bngsim._exceptions import ConversionError
    from bngsim.convert import net_to_sbml

    if not args.net.exists():
        print(f"error: no such file: {args.net}", file=sys.stderr)
        return 2

    out = args.out if args.out is not None else args.net.with_suffix(".xml")

    # Use the source .bngl for the L3 horizon: explicit --bngl, else a sibling.
    bngl = args.bngl
    if bngl is None:
        sibling = args.net.with_suffix(".bngl")
        if sibling.is_file():
            bngl = sibling
    elif not bngl.is_file():
        print(f"error: no such file: {bngl}", file=sys.stderr)
        return 2

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = net_to_sbml(
                args.net,
                out,
                validate=_resolve_gate(args),
                strict=not args.allow_lossy,
                bngl=bngl,
                t_span=(0.0, args.t_end),
                n_points=args.n_points,
            )
            sidecar_path = None
            if args.sidecar:
                sidecar_path = Path(out).with_suffix(".sedml")
                if report.protocol is not None and not report.protocol.is_empty:
                    # The real .bngl protocol (every simulate/scan).
                    from bngsim.convert import write_sedml_protocol

                    write_sedml_protocol(
                        report.protocol, sidecar_path, model_source=str(out)
                    )
                else:
                    # No real protocol — fabricate a default, but warn + mark it.
                    from bngsim._exceptions import ConversionWarning
                    from bngsim.convert import default_protocol, write_sedml

                    why = (
                        "the .bngl carried no simulate action" if bngl is not None
                        else "no .bngl protocol source was supplied"
                    )
                    warnings.warn(
                        f"no simulation protocol available ({why}); the SED-ML "
                        "sidecar is a bngsim-generated DEFAULT (NOT the modeller's "
                        "protocol)",
                        ConversionWarning,
                        stacklevel=2,
                    )
                    proto = default_protocol(
                        args.net, model_source=str(out), model_format="sbml"
                    )
                    write_sedml(proto, sidecar_path, synthesized_default=True)
    except ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    for w in caught:
        print(f"warning: {w.message}", file=sys.stderr)

    if not args.quiet:
        print(report.summary())
        if sidecar_path is not None:
            print(f"  sidecar: {sidecar_path} (SED-ML uniform time course)")

    return 0 if report.ok else 1


def _build_sbml2bngl_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bngsim-sbml2bngl",
        description=(
            "Convert an SBML model to a compartmental BioNetGen .bngl model block "
            "(cBNGL — recovers static compartment volumes, GH #224). Unlike "
            "sbml2net's flat unit-volume .net, the emitted model carries a "
            "begin compartments block so non-unit-volume models round-trip "
            "faithfully through BNG2.pl generate_network. The output is the "
            "begin model … end model block (no actions yet — the events→actions "
            "protocol channel is #224 phase 2); append a generate_network action "
            "to run it through BNG2.pl."
        ),
    )
    p.add_argument("sbml", type=Path, help="source SBML .xml file")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output .bngl path (default: alongside the input with a .bngl suffix)",
    )
    p.add_argument(
        "--allow-lossy",
        action="store_true",
        help=(
            "emit a best-effort model even when it uses constructs the cBNGL "
            "writer cannot carry faithfully yet (events, live/time-varying "
            "volumes, cross-compartment/transport reactions, Michaelis–Menten "
            "kinetics); downgrades the error to a warning"
        ),
    )
    p.add_argument(
        "--gate",
        action="store_true",
        help=(
            "additionally prove faithfulness by round-tripping the emitted .bngl "
            "through BNG2.pl generate_network and comparing the ODE right-hand side "
            "to the source (validate='bng2'); needs BNG2.pl on $BNGPATH or PATH. "
            "Fails (exit 1) if the round-trip is not RHS-faithful"
        ),
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the per-conversion summary (still prints errors)",
    )
    return p


def sbml2bngl_main(argv: list[str] | None = None) -> int:
    args = _build_sbml2bngl_parser().parse_args(argv)
    # Import here so ``--help`` stays fast and import errors surface cleanly.
    from bngsim._exceptions import ConversionError
    from bngsim.convert import sbml_to_bngl

    if not args.sbml.exists():
        print(f"error: no such file: {args.sbml}", file=sys.stderr)
        return 2

    out = args.out if args.out is not None else args.sbml.with_suffix(".bngl")

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = sbml_to_bngl(
                args.sbml,
                out,
                strict=not args.allow_lossy,
                validate="bng2" if args.gate else None,
            )
    except ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    for w in caught:
        print(f"warning: {w.message}", file=sys.stderr)

    if not args.quiet:
        print(report.summary())

    # ``ok`` reflects the capability check, plus the BNG2.pl round-trip verdict when
    # --gate ran (the authoritative cBNGL faithfulness check; no in-tree reader).
    return 0 if report.ok else 1


def _build_validate_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bngsim-validate-conversion",
        description=(
            "Validate a format conversion (SBML⇄.net) at levels L0–L4 (GH #217): "
            "L0 syntactic validity, L1 structural equivalence, L2 round-trip "
            "identity, L3 numerical equivalence (hard gates) and L4 symbolic "
            "equivalence (best-effort, non-gating). The direction is inferred "
            "from the source suffix (.xml/.sbml or .net)."
        ),
    )
    p.add_argument("source", type=Path, help="source model (.xml/.sbml or .net)")
    p.add_argument(
        "--direction",
        choices=("sbml2net", "net2sbml"),
        default=None,
        help="conversion direction (default: inferred from the source suffix)",
    )
    p.add_argument(
        "--levels",
        default="all",
        help=(
            "comma-separated subset of L0,L1,L2,L3,L4 to run (default: all)"
        ),
    )
    p.add_argument(
        "--allow-lossy",
        action="store_true",
        help="emit a best-effort conversion instead of refusing unfaithful constructs",
    )
    p.add_argument(
        "--t-end",
        type=float,
        default=100.0,
        help="end time for the L3 simulation comparison (default: 100)",
    )
    p.add_argument(
        "--n-points",
        type=int,
        default=101,
        help="number of time points for the L3 comparison (default: 101)",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="keep the converted artifacts in this directory (default: temp, cleaned up)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit the structured report as JSON instead of the prose summary",
    )
    return p


def validate_main(argv: list[str] | None = None) -> int:
    args = _build_validate_parser().parse_args(argv)
    from bngsim.convert import validate_conversion

    if not args.source.exists():
        print(f"error: no such file: {args.source}", file=sys.stderr)
        return 2

    levels = "all" if args.levels == "all" else tuple(
        s.strip() for s in args.levels.split(",") if s.strip()
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_conversion(
            args.source,
            direction=args.direction,
            levels=levels,
            strict=not args.allow_lossy,
            t_span=(0.0, args.t_end),
            n_points=args.n_points,
            out_dir=args.out_dir,
        )

    for w in caught:
        print(f"warning: {w.message}", file=sys.stderr)

    if args.json:
        import json

        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    # Non-zero exit when any hard gate failed, so scripts/CI can gate on it.
    return 0 if report.ok else 1


def _build_omex_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bngsim-omex",
        description=(
            "Pack/unpack a COMBINE archive (.omex) — the standard zip container "
            "bundling SBML + SED-ML + a manifest.xml (GH #219). Container plumbing "
            "only; the model/protocol semantics come from the sbml/net/sedml channels."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pack = sub.add_parser(
        "pack",
        help="bundle a .net (→ SBML + derived SED-ML) into a .omex archive",
        description=(
            "Convert a BioNetGen .net to SBML, derive a SED-ML simulation protocol "
            "(uniform time course over every observable), and bundle both plus a "
            "manifest.xml into one .omex archive."
        ),
    )
    pack.add_argument("net", type=Path, help="source BioNetGen .net file")
    pack.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output .omex path (default: alongside the input with a .omex suffix)",
    )
    pack.add_argument(
        "--bngl",
        type=Path,
        default=None,
        help=(
            "the source .bngl the .net was generated from. Its WHOLE simulate "
            "protocol (every simulate/parameter_scan + overrides) is carried into "
            "the SED-ML, and its horizon drives the gate's L3. Auto-detected as a "
            "sibling <stem>.bngl when present and not given."
        ),
    )
    pack.add_argument(
        "--gate",
        choices=("none", "L1", "full"),
        default="full",
        help=(
            "validation gate on the .net→SBML conversion before packaging "
            "(default: full — the OMEX ships the L0–L4 verdict; exits non-zero on "
            "a hard-gate failure)"
        ),
    )
    pack.add_argument(
        "--t-end", type=float, default=100.0, help="protocol/L3-fallback end time (default: 100)"
    )
    pack.add_argument(
        "--n-points", type=int, default=101, help="protocol/L3-fallback time points (default: 101)"
    )
    pack.add_argument(
        "--allow-lossy",
        action="store_true",
        help="emit best-effort SBML for constructs it cannot carry faithfully",
    )
    pack.add_argument(
        "--no-source",
        action="store_true",
        help=(
            "do not bundle the original source files (the .net, and the .bngl when "
            "given) into the archive. By default they ride along as provenance — the "
            "SBML stays the master/curated entry — so a published archive carries the "
            "modeller's rule-based formulation, not just the flattened SBML"
        ),
    )
    pack.add_argument(
        "--no-provenance",
        action="store_true",
        help=(
            "do not record provenance. By default the archive carries a COMBINE "
            "metadata.rdf (creator=bngsim version, date) and a bngsim-conversion.json "
            "with the faithfulness verdict (gate, L0–L4, ok), so the verified-faithful "
            "claim is auditable from inside the archive"
        ),
    )
    pack.add_argument(
        "-q", "--quiet", action="store_true", help="suppress the summary (still prints errors)"
    )

    unpack = sub.add_parser(
        "unpack",
        help="extract a .omex archive and report its model + protocol",
        description=(
            "Unzip a .omex archive, parse its manifest, and dispatch the master "
            "model (SBML/.net) and SED-ML entries to the bngsim readers."
        ),
    )
    unpack.add_argument("omex", type=Path, help="source .omex archive")
    unpack.add_argument(
        "-d",
        "--extract-dir",
        type=Path,
        default=None,
        help="directory to extract into (default: a temp dir, kept for inspection)",
    )
    unpack.add_argument(
        "-q", "--quiet", action="store_true", help="suppress the summary (still prints errors)"
    )

    tonet = sub.add_parser(
        "to-net",
        help="convert a .omex archive's SBML model to a BioNetGen .net (reverse of pack)",
        description=(
            "Read the archive's master SBML model and its SED-ML protocol, convert "
            "the SBML to a .net, and use the carried protocol's horizon to drive the "
            "gate's L3 check — the reverse of `pack`."
        ),
    )
    tonet.add_argument("omex", type=Path, help="source .omex archive")
    tonet.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output .net path (default: alongside the input with a .net suffix)",
    )
    tonet.add_argument(
        "--gate",
        choices=("none", "L1", "full"),
        default="full",
        help=(
            "validation gate on the SBML→.net conversion (default: full — the OMEX "
            "is the verified-faithful container, so the unpack ships the L0–L4 "
            "verdict; exits non-zero on a hard-gate failure)"
        ),
    )
    tonet.add_argument(
        "--t-end", type=float, default=100.0, help="L3-fallback end time when the archive carries no horizon (default: 100)"
    )
    tonet.add_argument(
        "--n-points", type=int, default=101, help="L3-fallback time points (default: 101)"
    )
    tonet.add_argument(
        "--allow-lossy",
        action="store_true",
        help="emit a best-effort network for constructs plain .net cannot carry faithfully",
    )
    tonet.add_argument(
        "--actions-out",
        type=Path,
        default=None,
        help=(
            "where to write the .bngl actions block composed from EVERY SED-ML "
            "experiment in the archive (GH #222; default: alongside the .net with "
            "a .bngl suffix)"
        ),
    )
    tonet.add_argument(
        "--no-actions",
        action="store_true",
        help="do not emit the .bngl actions block (only the .net network)",
    )
    tonet.add_argument(
        "-q", "--quiet", action="store_true", help="suppress the summary (still prints errors)"
    )
    return p


def omex_main(argv: list[str] | None = None) -> int:
    args = _build_omex_parser().parse_args(argv)
    from bngsim._exceptions import ConversionError

    if args.cmd == "pack":
        from bngsim.convert import net_to_omex

        if not args.net.exists():
            print(f"error: no such file: {args.net}", file=sys.stderr)
            return 2
        out = args.out if args.out is not None else args.net.with_suffix(".omex")

        bngl = args.bngl
        if bngl is None:
            sibling = args.net.with_suffix(".bngl")
            if sibling.is_file():
                bngl = sibling
        elif not bngl.is_file():
            print(f"error: no such file: {bngl}", file=sys.stderr)
            return 2

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                report = net_to_omex(
                    args.net,
                    out,
                    bngl=bngl,
                    gate=args.gate,
                    include_source=not args.no_source,
                    provenance=not args.no_provenance,
                    t_span=(0.0, args.t_end),
                    n_points=args.n_points,
                    strict=not args.allow_lossy,
                )
        except ConversionError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        for w in caught:
            print(f"warning: {w.message}", file=sys.stderr)
        if not args.quiet:
            print(report.summary())
            print(f"  archive: {out} (SBML + SED-ML + manifest.xml)")
        # Non-zero exit when a hard gate failed, so scripts/CI can gate on it.
        return 0 if report.ok else 1

    if args.cmd == "to-net":
        from bngsim.convert import omex_to_net

        if not args.omex.exists():
            print(f"error: no such file: {args.omex}", file=sys.stderr)
            return 2
        out = args.out if args.out is not None else args.omex.with_suffix(".net")

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                report = omex_to_net(
                    args.omex,
                    out,
                    gate=args.gate,
                    t_span=(0.0, args.t_end),
                    n_points=args.n_points,
                    strict=not args.allow_lossy,
                    actions_out=args.actions_out,
                    write_actions=not args.no_actions,
                )
        except ConversionError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        for w in caught:
            print(f"warning: {w.message}", file=sys.stderr)
        if not args.quiet:
            print(report.summary())
            print(f"  network: {out}")
        # Non-zero exit when a hard gate failed, so scripts/CI can gate on it.
        return 0 if report.ok else 1

    # unpack
    from bngsim.convert import read_omex

    if not args.omex.exists():
        print(f"error: no such file: {args.omex}", file=sys.stderr)
        return 2
    try:
        archive = read_omex(args.omex, extract_dir=args.extract_dir)
    except ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(archive.summary())
        model_entry = archive.master_model_entry()
        if model_entry is not None:
            print(f"  model:    {model_entry.location} ({model_entry.kind})")
        sed_entries = archive.sedml_entries()
        sed_entry = archive.master_sedml_entry()
        if sed_entry is not None:
            extra = (
                f" (+ {len(sed_entries) - 1} more SED-ML file"
                f"{'' if len(sed_entries) == 2 else 's'}; "
                "omex to-net composes every experiment)"
                if len(sed_entries) > 1 else ""
            )
            print(f"  protocol: {sed_entry.location} (SED-ML){extra}")
        else:
            print("  protocol: none (would derive a default uniform time course)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
