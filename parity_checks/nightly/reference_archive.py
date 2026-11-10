#!/usr/bin/env python3
"""The nightly's reference archive: what lets a night compare bngsim against the
reference engines without running them (issue #805).

Each oracle-backed parity runner can record its reference engine's side of every
job (``--write-reference-archive DIR``, one ``.npz`` per job) and replay a sweep
from that record (``--reference-archive DIR``) with the engine not installed. An
entry that no longer describes its job is refused, never compared
(``archive_stale``). This file is the plumbing around those directories, for
``.github/workflows/nightly-parity.yml``:

``prepare SUITE --mode MODE --root DIR``
    the ``KEY=value`` lines a nightly job appends to ``$GITHUB_ENV``:
    ``ARCHIVE_FLAGS`` for the runner and ``ARCHIVE_DIR``. ``replay`` fetches the
    suite's asset into ``DIR/SUITE`` first; ``record`` hands the runner that
    empty directory to write; ``live`` passes nothing. ``--archive FILE`` has a
    replay read the asset from FILE instead of downloading it.
``fetch SUITE --dest DIR [--archive FILE]``
    download the suite's asset from the release the committed manifest
    (``reference_archive.json``) names, or read it from FILE, check its sha256
    against that manifest, and write its entries into DIR. Nothing else in the
    asset becomes a file.
``locate SUITE``
    the suite's asset as ``url=``, ``sha256=`` and ``name=`` lines, for
    ``$GITHUB_OUTPUT``: what ``.github/actions/reference-side`` hands
    ``cached-download``, so a replay reads the asset from the Actions cache and a
    GitHub release-download outage does not stop it (issue #1607).
``pack SUITE --dir DIR --report R[,R...] --out-dir OUT``
    pack a recorded directory into ``OUT/reference-SUITE.tar.gz`` (deterministic:
    fixed order, times, owners and modes) and a ``.json`` sidecar describing it.
    Every PASS row of the run's report must have a recorded entry, or nothing is
    packed: an archive missing one would turn that good case into an
    ``archive_stale`` alert every night. A TIMEOUT row with no entry is listed
    (``timed_out_unrecorded``) for the same reason: a replay that needs the entry
    refuses it, which alerts the night after a live night that passes it, until
    the committed baseline records the row that way (issue #1599). An
    entry goes in only if its model's
    license, as a corpus provenance manifest records it, allows a copy; the rest
    are listed and left out, and their nightly row is ``archive_stale`` until the
    license is reviewed.
``manifest --release TAG --sidecars OUT --out FILE``
    the manifest to commit once the assets are published: one record per suite,
    in the form of ``parity_checks/manifest.schema.json``.

Recording, publishing and committing a new archive is in ``README.md``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PC = HERE.parent  # parity_checks/
REPO = PC.parent
MANIFEST = HERE / "reference_archive.json"
RELEASE_URL = "https://github.com/lanl/bngsim/releases/download/{tag}/{asset}"

sys.path.insert(0, str(PC / "rr_parity"))
from fetch_biomodels_mirror import _download  # noqa: E402

# Every nightly suite whose reference is an engine run, and that engine. Both BNG
# ODE arms replay the bng_ode archive: run_network's side does not depend on how
# bngsim evaluates its own right-hand side.
SUITES = {
    "rr_ode": "roadrunner",
    "bng_ode": "bng",
    "bng_ssa": "bng",
    "bng_nf": "bng",
    "amici_sens": "amici",
    "rr_ssa": "roadrunner",
}

# Where a model's source and license are recorded: every model a suite archives
# is in one of these (BioModels by accession, the BNG corpus by vendored path).
PROVENANCE = (
    REPO / "benchmarks" / "suites" / "biomodels" / "biomodels_provenance.json",
    PC / "bng_parity" / "manifest.json",
)

# Licenses that allow a copy of what was computed from a model, given the notices
# below. Anything else, "unknown" included, keeps the model's entries out of a
# published asset until someone reviews it (issue #805's licensing rule).
REDISTRIBUTABLE = frozenset({"CC0-1.0", "CC-BY-4.0", "MIT"})

# What travels with an asset holding anything but CC0 entries: NOTICE carries the
# attribution CC-BY-4.0 requires and the MIT text, and points at RuleMonkey's own
# license, which goes along at the path NOTICE names.
NOTICES = ("NOTICE", "third_party/rulemonkey/LICENSE")

# An entry's file name, as every runner's _archive_file writes it (ARCHIVE_FORMAT 2):
# the sanitized key and 12 hex digits of its sha256.
_ENTRY = re.compile(r"[A-Za-z0-9._-]+-[0-9a-f]{12}\.npz")

README = """\
Reference-engine outputs for bngsim's nightly parity check: suite {suite},
reference engine {engine} (bngsim issue #805).

One .npz per job: what the reference engine returned, or the error it raised,
with the job identity the entry is valid for. Recorded by
.github/workflows/nightly-parity.yml in record mode and read back by the suite's
runner with --reference-archive, which refuses an entry that no longer matches
its job.

These are outputs computed from the models entries.json lists, each with its
upstream source and license, and those licenses apply to them: BioModels are
CC0-1.0, RuleHub and RuleMonkey models MIT, BNGL-Models CC-BY-4.0. Where any
entry is not CC0, NOTICE (bngsim's, with the attribution and license texts) and
the files it names come along.

Checked and unpacked by parity_checks/nightly/reference_archive.py against the
sha256 in parity_checks/nightly/reference_archive.json.
"""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _licenses() -> dict[str, tuple[str, str]]:
    """model id -> (source, SPDX license), from the corpus provenance manifests."""
    out: dict[str, tuple[str, str]] = {}
    for path in PROVENANCE:
        for r in json.loads(path.read_text())["records"]:
            out[r["id"]] = (r["source"], (r.get("license") or {}).get("spdx") or "unknown")
    return out


def _record(manifest: Path, suite: str) -> dict:
    if not manifest.is_file():
        raise SystemExit(f"{suite}: no reference archive manifest at {manifest}")
    for r in json.loads(manifest.read_text())["records"]:
        if r["id"] == suite:
            return r
    raise SystemExit(f"{suite}: {manifest.name} holds no asset for this suite")


# ─── fetch ──────────────────────────────────────────────────────────────────


def locate(suite: str, manifest: Path = MANIFEST) -> list[str]:
    """The suite's asset as ``$GITHUB_OUTPUT`` lines: its URL, sha256 and file name."""
    rec = _record(manifest, suite)
    url = rec["origin"]["url"]
    return [f"url={url}", f"sha256={rec['sha256']}", f"name={url.rsplit('/', 1)[1]}"]


def _read(path: Path, tmp) -> str:
    """Copy ``path`` into ``tmp``; the sha256 of what was copied."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
            tmp.write(chunk)
    return digest.hexdigest()


def fetch(suite: str, dest: Path, manifest: Path = MANIFEST, archive: Path | None = None) -> int:
    """Download, check and unpack the suite's asset into ``dest``, or read it
    from ``archive``, a copy of it, checked the same way. Returns the number of
    entries written; any shortfall or mismatch is a SystemExit, so a night that
    cannot replay fails instead of comparing nothing."""
    rec = _record(manifest, suite)
    url = rec["origin"]["url"]
    source = url if archive is None else str(archive)
    prefix = f"reference-{suite}/"
    n = 0
    with tempfile.TemporaryFile() as tmp:
        if archive is None:
            got = _download(url, tmp)
            if got is None:
                raise SystemExit(f"{suite}: could not download {url}")
        elif not archive.is_file():
            raise SystemExit(f"{suite}: {archive} is not a file")
        else:
            got = _read(archive, tmp)
        if got != rec["sha256"]:
            raise SystemExit(
                f"{suite}: {source} has sha256 {got}; the manifest pins {rec['sha256']}"
            )
        tmp.seek(0)
        dest.mkdir(parents=True, exist_ok=True)
        # Only an entry's own file name, checked against the pattern every runner
        # writes, becomes a path, and only under dest.
        with tarfile.open(fileobj=tmp, mode="r:gz") as tar:
            for info in tar:
                if not (info.isfile() and info.name.startswith(prefix)):
                    continue
                name = info.name[len(prefix) :]
                if not _ENTRY.fullmatch(name):
                    continue
                (dest / name).write_bytes(tar.extractfile(info).read())
                n += 1
    if n != rec.get("n_entries"):
        raise SystemExit(
            f"{suite}: {n} entries unpacked; the manifest records {rec.get('n_entries')}"
        )
    print(f"{suite}: {n} reference entries from {source}, sha256 checked", file=sys.stderr)
    return n


def prepare(
    suite: str, mode: str, root: Path, manifest: Path = MANIFEST, archive: Path | None = None
) -> list[str]:
    """The ``$GITHUB_ENV`` lines that point a nightly job's runner at its archive.
    A replay reads the asset from ``archive`` when one is given."""
    d = root / suite
    if mode == "live":
        return ["ARCHIVE_FLAGS=", "ARCHIVE_DIR="]
    if mode == "record":
        d.mkdir(parents=True, exist_ok=True)
        return [f"ARCHIVE_FLAGS=--write-reference-archive {d}", f"ARCHIVE_DIR={d}"]
    if mode == "replay":
        fetch(suite, d, manifest, archive)
        return [f"ARCHIVE_FLAGS=--reference-archive {d}", f"ARCHIVE_DIR={d}"]
    raise SystemExit(f"unknown mode {mode!r}: replay, live or record")


# ─── pack ───────────────────────────────────────────────────────────────────


def _entry_meta(path: Path) -> dict:
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        return json.loads(bytes(z["meta"]).decode())


def _tar_member(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = size, 0, 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _unrecorded(
    reports: list[Path], recorded: set[tuple[str, str | None]]
) -> list[tuple[str, str]]:
    """``(case, outcome)`` for each row of ``reports`` with no recorded entry. An
    entry that names no method (bng_stoch's, ssa_screen's) covers every row of its
    model."""
    missing = []
    for path in reports:
        for row in json.loads(path.read_text())["results"]:
            mid, method = row["model_id"], row.get("method")
            if (mid, method) not in recorded and (mid, None) not in recorded:
                missing.append((f"{mid}|{method}", row["outcome"]))
    return sorted(missing)


def _good_in_baseline_only(reports: list[Path], baseline: Path) -> list[str]:
    """Cases good in the committed baseline that were not PASS in the recorded run.
    Their entries hold whatever path the recorded run took (an SSA screen row that
    timed out into a partial-horizon ladder records only that ladder), so a night
    that takes the baseline's path may find what it needs missing. Listed, not
    refused: one runner-speed flake must not make a record impossible to publish."""
    good = {
        c for c, v in json.loads(baseline.read_text())["cases"].items() if v["outcome"] == "PASS"
    }
    passed = {
        f"{row['model_id']}|{row.get('method', '')}"
        for path in reports
        for row in json.loads(path.read_text())["results"]
        if row["outcome"] == "PASS"
    }
    return sorted(good - passed)


def pack(
    suite: str,
    src: Path,
    out_dir: Path,
    reports: list[Path] | None = None,
    baseline: Path | None = None,
) -> dict:
    """Pack ``src`` into ``out_dir/reference-SUITE.tar.gz`` and write its sidecar."""
    if suite not in SUITES:
        raise SystemExit(f"unknown suite {suite!r}: {', '.join(SUITES)}")
    licenses = _licenses()
    entries, excluded = [], []
    recorded: set[tuple[str, str | None]] = set()
    for f in sorted(src.glob("*.npz")):
        if not _ENTRY.fullmatch(f.name):  # e.g. a .tmp.npz a killed worker left behind
            continue
        try:
            meta = _entry_meta(f)
        except Exception as exc:
            excluded.append({"file": f.name, "model_id": None, "reason": f"unreadable: {exc}"})
            continue
        mid = meta.get("model_id") or meta.get("name")
        recorded.add((mid, meta.get("method")))
        source, spdx = licenses.get(mid, (None, "unknown"))
        if spdx not in REDISTRIBUTABLE:
            excluded.append({"file": f.name, "model_id": mid, "reason": f"license {spdx}"})
            continue
        entries.append(
            {
                "file": f.name,
                "model_id": mid,
                "source": source,
                "license": spdx,
                "reference_version": meta.get("reference_version"),
            }
        )
    if not entries:
        raise SystemExit(f"{suite}: no entry to pack in {src}")
    unrecorded = _unrecorded(reports or [], recorded)
    missing = [case for case, outcome in unrecorded if outcome == "PASS"]
    if missing:
        raise SystemExit(
            f"{suite}: {len(missing)} PASS row(s) have no recorded reference, so a replay "
            f"would refuse them every night; not packing: {missing[:10]}"
        )

    spdx = sorted({e["license"] for e in entries})
    notices = [] if spdx == ["CC0-1.0"] else list(NOTICES)
    top = f"reference-{suite}"
    small = [
        ("README.txt", README.format(suite=suite, engine=SUITES[suite]).encode()),
        ("entries.json", (json.dumps(entries, indent=1) + "\n").encode()),
        *((n, (REPO / n).read_bytes()) for n in notices),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    asset = out_dir / f"{top}.tar.gz"
    with (
        open(asset, "wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for name, data in small:
            tar.addfile(_tar_member(f"{top}/{name}", len(data)), io.BytesIO(data))
        for e in entries:
            path = src / e["file"]
            with open(path, "rb") as fh:
                tar.addfile(_tar_member(f"{top}/{e['file']}", path.stat().st_size), fh)

    sidecar = {
        "suite": suite,
        "asset": asset.name,
        "sha256": _sha256_file(asset),
        "bytes": asset.stat().st_size,
        "n_entries": len(entries),
        "excluded": excluded,
        "licenses": spdx,
        "reference_engine": SUITES[suite],
        "reference_versions": sorted({str(v) for e in entries if (v := e["reference_version"])}),
        "good_in_baseline_only": (
            _good_in_baseline_only(reports or [], baseline) if baseline is not None else []
        ),
        # Jobs killed before their reference was saved (issue #1290). A replay that
        # needs the entry refuses them as archive_stale, an alert on the night after
        # a live night that passes them until the committed baseline records them
        # that way (#1599). Listed, not refused: a model too slow for
        # the recording must not block the rest. TIMEOUT rows only: a row of another
        # outcome may need no entry (ssa_screen decides its time-event rows before
        # it reads the archive).
        "timed_out_unrecorded": [case for case, outcome in unrecorded if outcome == "TIMEOUT"],
    }
    (out_dir / f"{top}.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    print(
        f"{asset.name}: {len(entries)} entries, {len(excluded)} left out, "
        f"{sidecar['bytes'] / 1e6:.1f} MB, sha256 {sidecar['sha256']}"
    )
    for x in excluded:
        print(f"  left out {x['model_id'] or x['file']}: {x['reason']}")
    if sidecar["good_in_baseline_only"]:
        cases = sidecar["good_in_baseline_only"]
        print(
            f"  {len(cases)} case(s) good in the baseline were not PASS in this recording; a "
            f"replay that takes the baseline's path may refuse them: {', '.join(cases[:20])}"
        )
    if sidecar["timed_out_unrecorded"]:
        cases = sidecar["timed_out_unrecorded"]
        print(
            f"  {len(cases)} case(s) timed out before their reference was recorded; a "
            f"replay that needs it refuses them as archive_stale: {', '.join(cases[:20])}"
        )
    return sidecar


# ─── manifest ───────────────────────────────────────────────────────────────


def manifest(
    release: str,
    sidecars: Path,
    out: Path,
    *,
    recorded_by: str | None = None,
    commit: str | None = None,
    today: str | None = None,
) -> dict:
    """The committed manifest for the assets of ``release``: every suite or none."""
    today = today or _dt.date.today().isoformat()
    records = []
    for suite in SUITES:
        side_path = sidecars / f"reference-{suite}.json"
        if not side_path.is_file():
            raise SystemExit(f"no {side_path.name} in {sidecars}: every suite or none")
        side = json.loads(side_path.read_text())
        records.append(
            {
                "id": suite,
                "source": "reference-archive",
                "origin": {
                    "kind": "url",
                    "url": RELEASE_URL.format(tag=release, asset=side["asset"]),
                    "retrieved": today,
                },
                "sha256": side["sha256"],
                "vendored": None,
                "license": {
                    "spdx": " AND ".join(side["licenses"]),
                    "notice": None if side["licenses"] == ["CC0-1.0"] else "NOTICE",
                },
                "reference_engine": side["reference_engine"],
                "reference_versions": side["reference_versions"],
                "n_entries": side["n_entries"],
                "n_excluded": len(side["excluded"]),
                "bytes": side["bytes"],
            }
        )
    doc = {
        "schema_version": "1.0",
        "corpus": "nightly/reference_archive",
        "generated": today,
        "upstream_pin": {"release": release, "recorded_by": recorded_by, "commit": commit},
        "records": records,
    }
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def caveats(sidecars: Path) -> list[str]:
    """Markdown bullets for the publish summary: per suite, the cases good in the
    baseline that were not PASS in the recording (see ``_good_in_baseline_only``),
    and the cases that timed out before their reference was recorded."""
    lines = []
    for suite in SUITES:
        side = json.loads((sidecars / f"reference-{suite}.json").read_text())
        cases = side.get("good_in_baseline_only") or []
        if cases:
            lines.append(
                f"- **{suite}**: {len(cases)} case(s) good in the baseline were not PASS in "
                f"this recording: {', '.join(cases[:30])}"
            )
        cases = side.get("timed_out_unrecorded") or []
        if cases:
            lines.append(
                f"- **{suite}**: {len(cases)} case(s) timed out before their reference was "
                f"recorded; a replay that needs it refuses them as archive_stale: "
                f"{', '.join(cases[:30])}"
            )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="$GITHUB_ENV lines for a nightly job")
    p.add_argument("suite", choices=sorted(SUITES))
    p.add_argument("--mode", required=True, choices=("replay", "live", "record"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--archive", type=Path, help="a replay reads the asset from this copy")
    p = sub.add_parser("fetch", help="download, check and unpack a suite's asset")
    p.add_argument("suite", choices=sorted(SUITES))
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--archive", type=Path, help="read the asset from this copy")
    p = sub.add_parser("locate", help="a suite's asset URL, sha256 and file name")
    p.add_argument("suite", choices=sorted(SUITES))
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p = sub.add_parser("pack", help="pack a recorded directory into an asset")
    p.add_argument("suite", choices=sorted(SUITES))
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--report", required=True, help="the run's report(s), comma-separated")
    p.add_argument("--baseline", type=Path, help="the suite's committed verdict baseline")
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("manifest", help="the manifest to commit for a published release")
    p.add_argument("--release", required=True)
    p.add_argument("--sidecars", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--recorded-by")
    p.add_argument("--commit")
    args = ap.parse_args()

    if args.cmd == "prepare":
        print("\n".join(prepare(args.suite, args.mode, args.root, args.manifest, args.archive)))
    elif args.cmd == "fetch":
        fetch(args.suite, args.dest, args.manifest, args.archive)
    elif args.cmd == "locate":
        print("\n".join(locate(args.suite, args.manifest)))
    elif args.cmd == "pack":
        reports = [Path(r) for r in args.report.split(",") if r]
        pack(args.suite, args.dir, args.out_dir, reports, args.baseline)
    else:
        manifest(
            args.release, args.sidecars, args.out, recorded_by=args.recorded_by, commit=args.commit
        )
        # The publish summary's caveats, on stdout.
        print("\n".join(caveats(args.sidecars)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
