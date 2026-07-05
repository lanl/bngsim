#!/usr/bin/env python3
"""Deep NFsim parity tracing: BNGsim in-process vs standalone NFsim CLI.

Compares *full sampled trajectories* and *event progression* using the same
XML + seed. The script writes machine-readable artifacts and reports the first
divergence point.

Outputs in --out-dir:
  - summary.json
  - bngsim_trace.tsv
  - nfsim_cli_trace.tsv
  - trace_diff.tsv
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = SCRIPT_DIR / "results"
# BioNetGen 2.9.3 install. Set BNGPATH to the install root; BNG2_PL / NFSIM
# override an individual tool. Default = canonical ~/Simulations install.
BNGPATH = os.environ.get("BNGPATH", os.path.expanduser("~/Simulations/BioNetGen-2.9.3"))
DEFAULT_BNG2_PL = Path(os.environ.get("BNG2_PL", os.path.join(BNGPATH, "BNG2.pl")))
DEFAULT_NFSIM_BIN = Path(os.environ.get("NFSIM", os.path.join(BNGPATH, "bin", "NFsim")))
EVENT_COL_ALIASES = {"eventcount", "event_count", "eventcounter", "event_counter"}


@dataclass(frozen=True)
class TraceTable:
    names: list[str]
    data: np.ndarray  # shape (n_rows, n_cols)


def parse_on_off(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"on", "true", "1", "yes"}:
        return True
    if normalized in {"off", "false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected on/off, got: {value!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Trace same-XML/same-seed NFsim parity between BNGsim and CLI "
            "(trajectory + event-count progression)."
        )
    )
    p.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Path to BNG XML file (preferred).",
    )
    p.add_argument(
        "--bngl",
        type=Path,
        default=None,
        help="Optional BNGL path; if set, script generates XML once via writeXML().",
    )
    p.add_argument("--bng2-pl", type=Path, default=DEFAULT_BNG2_PL)
    p.add_argument("--nfsim-bin", type=Path, default=DEFAULT_NFSIM_BIN)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--t-end", type=float, default=20.0)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--gml", type=int, default=1_000_000)
    p.add_argument("--timeout-s", type=float, default=300.0)
    p.add_argument("--time-atol", type=float, default=1e-12)
    p.add_argument("--obs-atol", type=float, default=0.0)
    p.add_argument("--obs-rtol", type=float, default=0.0)
    p.add_argument(
        "--wrapper-connectivity",
        type=parse_on_off,
        default=None,
        help="Override wrapper connectivity inference (on/off). Default: use wrapper default.",
    )
    p.add_argument(
        "--cli-connectivity",
        type=parse_on_off,
        default=False,
        help="Pass -connect to standalone NFsim CLI (on/off). Default: off.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: diagnostics/results/<timestamp>_nf_parity_trace",
    )
    p.add_argument("--tag", default="")
    args = p.parse_args()

    if (args.xml is None) == (args.bngl is None):
        raise SystemExit("Specify exactly one of --xml or --bngl.")
    if args.n_steps <= 0:
        raise SystemExit("--n-steps must be > 0.")
    if args.t_end <= 0:
        raise SystemExit("--t-end must be > 0.")
    return args


def strip_actions(content: str) -> str:
    pat = re.compile(
        r"^\s*(generate_network|simulate|simulate_nf|simulate_ssa|simulate_ode|"
        r"writeXML|writeNetwork|writeSBML|writeMDL|writeMfile|writeMexfile|"
        r"resetConcentrations|resetParameters|"
        r"saveConcentrations|saveParameters|"
        r"setConcentration|setParameter|"
        r"parameter_scan|bifurcate|"
        r"begin\s+actions|end\s+actions)\b.*$",
        re.IGNORECASE,
    )
    return "\n".join(ln for ln in content.splitlines() if not pat.match(ln.strip()))


def run_checked(cmd: list[str], cwd: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        msg = (
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout[-1000:]}\n"
            f"stderr:\n{proc.stderr[-1000:]}"
        )
        raise RuntimeError(msg)
    return proc


def make_out_dir(out_dir: Path | None, tag: str) -> Path:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir.resolve()
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    out = DEFAULT_OUT_ROOT / f"{stamp}_nf_parity_trace{suffix}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def resolve_xml(
    args: argparse.Namespace, out_dir: Path
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if args.xml is not None:
        xml = args.xml.resolve()
        if not xml.exists():
            raise FileNotFoundError(f"XML not found: {xml}")
        shutil.copy2(xml, out_dir / xml.name)
        return xml, None

    bngl = args.bngl.resolve()
    if not bngl.exists():
        raise FileNotFoundError(f"BNGL not found: {bngl}")
    if not args.bng2_pl.exists():
        raise FileNotFoundError(f"BNG2.pl not found: {args.bng2_pl}")

    td = tempfile.TemporaryDirectory(prefix="nf_parity_xml_")
    td_path = Path(td.name)
    clean = strip_actions(bngl.read_text()).rstrip()
    xml_input = td_path / f"{bngl.stem}_xml.bngl"
    xml_input.write_text(clean + "\n\nwriteXML()\n")

    run_checked(["perl", str(args.bng2_pl), str(xml_input)], td_path, args.timeout_s)
    xml = td_path / f"{bngl.stem}_xml.xml"
    if not xml.exists():
        raise RuntimeError(f"writeXML() did not produce expected file: {xml}")
    shutil.copy2(xml, out_dir / xml.name)
    return xml, td


def parse_numeric_table(path: Path) -> TraceTable:
    names: list[str] | None = None
    rows: list[list[float]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            fields = line.lstrip("#").replace(",", " ").split()
            if fields:
                names = fields
            continue
        fields = line.replace(",", " ").split()
        rows.append([float(x) for x in fields])

    if not rows:
        raise RuntimeError(f"No data rows found in table: {path}")
    data = np.asarray(rows, dtype=float)

    if names is None:
        names = [f"col_{i}" for i in range(data.shape[1])]
    if len(names) != data.shape[1]:
        names = [f"col_{i}" for i in range(data.shape[1])]
    return TraceTable(names=names, data=data)


def write_tsv(path: Path, names: list[str], data: np.ndarray) -> None:
    with path.open("w") as f:
        f.write("\t".join(names) + "\n")
        for row in data:
            f.write("\t".join(f"{float(v):.17g}" for v in row) + "\n")


def find_col(names: Iterable[str], aliases: set[str]) -> int | None:
    for idx, name in enumerate(names):
        if name.strip().lower() in aliases:
            return idx
    return None


def run_nfsim_cli_trace(
    nfsim_bin: Path,
    xml_path: Path,
    seed: int,
    t_end: float,
    n_steps: int,
    gml: int,
    cli_connectivity: bool,
    timeout_s: float,
    work_dir: Path,
) -> tuple[TraceTable, list[str]]:
    if not nfsim_bin.exists():
        raise FileNotFoundError(f"NFsim binary not found: {nfsim_bin}")
    out = work_dir / "nfsim_cli_trace.gdat"
    cmd = [
        str(nfsim_bin),
        "-xml",
        str(xml_path),
        "-sim",
        str(t_end),
        "-oSteps",
        str(n_steps),
        "-seed",
        str(seed),
        "-gml",
        str(gml),
        "-oec",
        "-cb",
        "-o",
        str(out),
    ]
    if cli_connectivity:
        cmd.append("-connect")
    run_checked(cmd, work_dir, timeout_s)
    if not out.exists():
        raise RuntimeError(f"NFsim CLI did not produce output file: {out}")
    return parse_numeric_table(out), cmd


def run_bngsim_trace(
    xml_path: Path,
    seed: int,
    t_end: float,
    n_steps: int,
    gml: int,
    wrapper_connectivity: bool | None,
) -> TraceTable:
    from bngsim._bngsim_core import NfsimSimulator

    sim = NfsimSimulator(str(xml_path))
    sim.set_molecule_limit(int(gml))
    if wrapper_connectivity is not None:
        sim.set_connectivity(bool(wrapper_connectivity))
    sim.initialize(int(seed))

    try:
        obs_names = list(sim.get_observable_names())
        n_obs = len(obs_names)
        times = np.linspace(0.0, float(t_end), int(n_steps) + 1, dtype=float)
        obs = np.empty((times.size, n_obs), dtype=float)
        event_counts = np.zeros(times.size, dtype=np.int64)

        obs[0, :] = np.asarray(sim.get_observable_values(), dtype=float)
        for i in range(1, times.size):
            seg = sim.simulate(float(times[i - 1]), float(times[i]), 2)
            seg_obs = np.asarray(seg.observable_data, dtype=float)
            obs[i, :] = seg_obs[-1, :]
            event_counts[i] = int(seg.solver_stats.n_steps)

        names = ["time"] + obs_names + ["EventCount"]
        data = np.column_stack([times, obs, event_counts.astype(float)])
        return TraceTable(names=names, data=data)
    finally:
        if sim.has_session():
            sim.destroy_session()


def compare_traces(
    bng: TraceTable,
    cli: TraceTable,
    time_atol: float,
    obs_atol: float,
    obs_rtol: float,
) -> tuple[dict, TraceTable]:
    b_time_idx = find_col(bng.names, {"time"})
    c_time_idx = find_col(cli.names, {"time"})
    b_event_idx = find_col(bng.names, EVENT_COL_ALIASES)
    c_event_idx = find_col(cli.names, EVENT_COL_ALIASES)

    if b_time_idx is None or c_time_idx is None:
        raise RuntimeError("Missing 'time' column in at least one trace.")
    if b_event_idx is None or c_event_idx is None:
        raise RuntimeError("Missing EventCount column in at least one trace.")

    b_obs_names = [nm for i, nm in enumerate(bng.names) if i not in {b_time_idx, b_event_idx}]
    c_obs_names = [nm for i, nm in enumerate(cli.names) if i not in {c_time_idx, c_event_idx}]
    common_obs = [nm for nm in b_obs_names if nm in c_obs_names]

    b_obs_idx = [bng.names.index(nm) for nm in common_obs]
    c_obs_idx = [cli.names.index(nm) for nm in common_obs]

    n_rows = min(bng.data.shape[0], cli.data.shape[0])
    row_count_match = bng.data.shape[0] == cli.data.shape[0]

    b_time = bng.data[:n_rows, b_time_idx]
    c_time = cli.data[:n_rows, c_time_idx]
    time_abs_diff = np.abs(b_time - c_time)
    time_bad = time_abs_diff > time_atol

    b_event = np.rint(bng.data[:n_rows, b_event_idx]).astype(np.int64)
    c_event = np.rint(cli.data[:n_rows, c_event_idx]).astype(np.int64)
    event_diff = b_event - c_event
    event_bad = event_diff != 0

    if common_obs:
        b_obs = bng.data[:n_rows, :][:, b_obs_idx]
        c_obs = cli.data[:n_rows, :][:, c_obs_idx]
        obs_abs_diff = np.abs(b_obs - c_obs)
        tol = obs_atol + obs_rtol * np.abs(c_obs)
        obs_bad = obs_abs_diff > tol
        safe = np.maximum(np.abs(c_obs), max(obs_atol, 1e-15))
        obs_rel_diff = obs_abs_diff / safe
        row_max_abs = np.max(obs_abs_diff, axis=1)
        row_max_rel = np.max(obs_rel_diff, axis=1)
        row_obs_bad_count = np.sum(obs_bad, axis=1)
    else:
        obs_bad = np.zeros((n_rows, 0), dtype=bool)
        row_max_abs = np.zeros(n_rows, dtype=float)
        row_max_rel = np.zeros(n_rows, dtype=float)
        row_obs_bad_count = np.zeros(n_rows, dtype=int)

    row_bad = time_bad | event_bad | (row_obs_bad_count > 0)

    def first_true(mask: np.ndarray) -> int | None:
        idx = np.flatnonzero(mask)
        return int(idx[0]) if idx.size > 0 else None

    first_row = first_true(row_bad)
    first_event = first_true(event_bad)
    first_time = first_true(time_bad)
    first_obs = None
    first_obs_name = None
    if common_obs:
        ij = np.argwhere(obs_bad)
        if ij.size > 0:
            first_obs = int(ij[0, 0])
            first_obs_name = common_obs[int(ij[0, 1])]

    b_event_delta = np.diff(b_event, prepend=b_event[:1])
    c_event_delta = np.diff(c_event, prepend=c_event[:1])
    event_delta_diff = b_event_delta - c_event_delta

    diff_names = [
        "time_bngsim",
        "time_cli",
        "event_bngsim",
        "event_cli",
        "event_delta_bngsim",
        "event_delta_cli",
        "event_delta_diff",
        "max_abs_obs_diff",
        "max_rel_obs_diff",
        "n_obs_mismatch",
    ]
    diff_data = np.column_stack(
        [
            b_time,
            c_time,
            b_event.astype(float),
            c_event.astype(float),
            b_event_delta.astype(float),
            c_event_delta.astype(float),
            event_delta_diff.astype(float),
            row_max_abs,
            row_max_rel,
            row_obs_bad_count.astype(float),
        ]
    )

    first_row_detail = None
    if first_row is not None:
        obs_mismatches: list[dict] = []
        if common_obs:
            for j, nm in enumerate(common_obs):
                if not obs_bad[first_row, j]:
                    continue
                obs_mismatches.append(
                    {
                        "name": nm,
                        "bngsim": float(b_obs[first_row, j]),
                        "cli": float(c_obs[first_row, j]),
                        "abs_diff": float(obs_abs_diff[first_row, j]),
                        "rel_diff": float(obs_rel_diff[first_row, j]),
                    }
                )

        first_row_detail = {
            "index": int(first_row),
            "time_bngsim": float(b_time[first_row]),
            "time_cli": float(c_time[first_row]),
            "event_bngsim": int(b_event[first_row]),
            "event_cli": int(c_event[first_row]),
            "event_delta_bngsim": int(b_event_delta[first_row]),
            "event_delta_cli": int(c_event_delta[first_row]),
            "time_abs_diff": float(time_abs_diff[first_row]),
            "observable_mismatches": obs_mismatches,
        }

    summary = {
        "row_count": int(n_rows),
        "row_count_match": bool(row_count_match),
        "bngsim_rows": int(bng.data.shape[0]),
        "cli_rows": int(cli.data.shape[0]),
        "common_observables": common_obs,
        "missing_in_bngsim": [n for n in c_obs_names if n not in b_obs_names],
        "missing_in_cli": [n for n in b_obs_names if n not in c_obs_names],
        "max_time_abs_diff": float(np.max(time_abs_diff)),
        "max_event_abs_diff": int(np.max(np.abs(event_diff))) if n_rows > 0 else 0,
        "max_obs_abs_diff": float(np.max(row_max_abs)) if n_rows > 0 else 0.0,
        "max_obs_rel_diff": float(np.max(row_max_rel)) if n_rows > 0 else 0.0,
        "first_row_divergence_index": first_row,
        "first_row_divergence_time_bngsim": float(b_time[first_row])
        if first_row is not None
        else None,
        "first_row_divergence_time_cli": float(c_time[first_row])
        if first_row is not None
        else None,
        "first_event_divergence_index": first_event,
        "first_event_divergence_time_bngsim": float(b_time[first_event])
        if first_event is not None
        else None,
        "first_event_divergence_time_cli": float(c_time[first_event])
        if first_event is not None
        else None,
        "first_time_divergence_index": first_time,
        "first_observable_divergence_index": first_obs,
        "first_observable_divergence_name": first_obs_name,
        "first_row_detail": first_row_detail,
        "parity_passed": first_row is None and row_count_match,
    }
    return summary, TraceTable(names=diff_names, data=diff_data)


def main() -> int:
    args = parse_args()
    out_dir = make_out_dir(args.out_dir, args.tag)
    run_dir = out_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    xml_temp: tempfile.TemporaryDirectory[str] | None = None
    try:
        xml_path, xml_temp = resolve_xml(args, out_dir)
        xml_copy_path = (out_dir / xml_path.name).resolve()
        bng_trace = run_bngsim_trace(
            xml_path=xml_path,
            seed=args.seed,
            t_end=args.t_end,
            n_steps=args.n_steps,
            gml=args.gml,
            wrapper_connectivity=args.wrapper_connectivity,
        )
        cli_trace, cli_cmd = run_nfsim_cli_trace(
            nfsim_bin=args.nfsim_bin.resolve(),
            xml_path=xml_path,
            seed=args.seed,
            t_end=args.t_end,
            n_steps=args.n_steps,
            gml=args.gml,
            cli_connectivity=bool(args.cli_connectivity),
            timeout_s=args.timeout_s,
            work_dir=run_dir,
        )
        comparison, diff_trace = compare_traces(
            bng_trace,
            cli_trace,
            time_atol=args.time_atol,
            obs_atol=args.obs_atol,
            obs_rtol=args.obs_rtol,
        )

        write_tsv(out_dir / "bngsim_trace.tsv", bng_trace.names, bng_trace.data)
        write_tsv(out_dir / "nfsim_cli_trace.tsv", cli_trace.names, cli_trace.data)
        write_tsv(out_dir / "trace_diff.tsv", diff_trace.names, diff_trace.data)

        summary = {
            "config": {
                "xml": str(xml_copy_path),
                "runtime_xml": str(xml_path.resolve()),
                "seed": int(args.seed),
                "t_end": float(args.t_end),
                "n_steps": int(args.n_steps),
                "gml": int(args.gml),
                "time_atol": float(args.time_atol),
                "obs_atol": float(args.obs_atol),
                "obs_rtol": float(args.obs_rtol),
                "wrapper_connectivity": args.wrapper_connectivity,
                "cli_connectivity": bool(args.cli_connectivity),
                "nfsim_bin": str(args.nfsim_bin.resolve()),
                "nfsim_cli_cmd": cli_cmd,
            },
            "comparison": comparison,
            "artifacts": {
                "bngsim_trace_tsv": str((out_dir / "bngsim_trace.tsv").resolve()),
                "nfsim_cli_trace_tsv": str((out_dir / "nfsim_cli_trace.tsv").resolve()),
                "trace_diff_tsv": str((out_dir / "trace_diff.tsv").resolve()),
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        print(f"[trace] output dir: {out_dir}")
        print(
            f"[trace] rows compared: {comparison['row_count']} "
            f"(row-count match={comparison['row_count_match']})"
        )
        print(
            f"[trace] common observables: {len(comparison['common_observables'])} "
            f"max_abs_obs_diff={comparison['max_obs_abs_diff']:.6g} "
            f"max_event_abs_diff={comparison['max_event_abs_diff']}"
        )
        if comparison["parity_passed"]:
            print("[trace] PASS: trajectory + event progression match within tolerances.")
        else:
            idx = comparison["first_row_divergence_index"]
            print(
                "[trace] DIVERGENCE:"
                f" first_row_idx={idx} "
                f"time_bngsim={comparison['first_row_divergence_time_bngsim']} "
                f"time_cli={comparison['first_row_divergence_time_cli']} "
                f"first_obs={comparison['first_observable_divergence_name']}"
            )
        return 0
    finally:
        if xml_temp is not None:
            xml_temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
