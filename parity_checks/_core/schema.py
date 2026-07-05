"""Spec (manifest) and results (report) schemas shared across parity suites.

The plan keeps three artifacts strictly separate:

  * manifest  — the SPEC: which model × method × reference engine to run, the
    run parameters, any per-model overrides (each with a reason), and the
    comparison oracle. Committed and stable; one job per line for clean diffs.
  * report    — the RESULTS of one run: per-job outcome, the metric value, the
    exception text, wall time, timestamp, and the engine versions used.
    Regenerated every run; never hand-edited.
  * golden    — per-job CHECKSUM + numeric FINGERPRINT (+ full trajectory for a
    representative subset) that consumers regenerate through their own bridge.

All three are plain JSON. Manifest and report use a compact layout: a
pretty-printed ``_meta`` header then one compact job object per line, so the
large-file hook stays happy and git diffs stay readable.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Manifest (spec)
# --------------------------------------------------------------------------- #
@dataclass
class Oracle:
    """How a job's two engines are compared, and the pass bar."""

    metric: str  # "max_rel_err" | "mean_zscore" | ... (see oracles.py)
    tol: float

    def to_dict(self) -> dict:
        return {"metric": self.metric, "tol": self.tol}


@dataclass
class Override:
    """A per-model deviation from defaults, applied identically to both engines.

    `reason` is mandatory: an override with no rationale is a silent fudge.
    Lifted out of the old hardcoded dicts (TEND_OVERRIDES, TOL_OVERRIDES, ...).
    """

    field: str  # e.g. "t_end", "atol", "symbol_rename"
    value: Any
    reason: str

    def to_dict(self) -> dict:
        return {"field": self.field, "value": self.value, "reason": self.reason}


@dataclass
class Job:
    """One unit of work: a model run by one method, compared to one reference.

    `model` is the suite-relative path to the vendored model file. `params`
    holds the suite-specific run configuration (time grid, tolerances, seed,
    n_rep, ...) — it stays free-form because the three suites differ; the
    fields above are the cross-suite contract.
    """

    model_id: str
    input_format: str  # "sbml" | "bngl" | "net"
    method: str  # "ode" | "ssa" | "nf" | "sens"
    reference_engine: str  # "roadrunner" | "bng" | "amici"
    model: str
    oracle: Oracle
    params: dict = field(default_factory=dict)
    overrides: list[Override] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "input_format": self.input_format,
            "method": self.method,
            "reference_engine": self.reference_engine,
            "model": self.model,
            "oracle": self.oracle.to_dict(),
            "params": self.params,
            "overrides": [o.to_dict() for o in self.overrides],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Job:
        return cls(
            model_id=d["model_id"],
            input_format=d["input_format"],
            method=d["method"],
            reference_engine=d["reference_engine"],
            model=d["model"],
            oracle=Oracle(**d["oracle"]),
            params=d.get("params", {}),
            overrides=[Override(**o) for o in d.get("overrides", [])],
            notes=d.get("notes", ""),
        )


# --------------------------------------------------------------------------- #
# Report (results)
# --------------------------------------------------------------------------- #
@dataclass
class JobResult:
    """The outcome of running one Job. Fields are blank/None when N/A."""

    model_id: str
    method: str
    reference_engine: str
    outcome: str  # Outcome value
    metric: str | None = None  # echo of the oracle metric
    value: float | None = None  # the actual max_rel / zscore observed
    tol: float | None = None
    exception: str = ""
    wall_sec: float | None = None
    timestamp: str = ""
    versions: dict = field(default_factory=dict)  # {"bngsim":..., "<ref>":...}
    comment: str = ""  # esoteric explanation (e.g. why a subtle case is all-clear)
    # Machine-readable sub-classification of WHY the reference engine refused,
    # set only on REFERENCE_FAILED rows (None otherwise). Splits the "reference
    # couldn't, bngsim ran" bucket into the settled-win vs still-needs-triage
    # subsets so a later run doesn't re-investigate decided cases — see each
    # suite's classifier for the vocabulary (e.g. rr_parity:
    # ``overstrict_missing_value`` = bngsim provably-correctly accepts a model the
    # reference over-strictly refused; ``feature_gap`` / ``integrator`` = still
    # unverified, triage-worthy). Defaulted/optional, so suites that don't
    # classify and older reports simply leave it null.
    reference_refusal: str | None = None
    # General machine-readable sub-classification of the OUTCOME, when one Outcome
    # value covers materially different cases a consumer must tell apart. Where
    # ``reference_refusal`` sub-classes only REFERENCE_FAILED (why the *reference*
    # refused), ``subclass`` is the open-vocabulary equivalent for any outcome —
    # e.g. the rr_parity SSA screen tags a DIFF as ``bngsim_suspect`` / ``ode_level``
    # (real, scoring) vs ``diff_not_bngsim`` / ``rr_known`` (oracle-attributed away
    # from bngsim, non-scoring). A suite's scoring policy decides which
    # (outcome, subclass) pairs are expected; see e.g. ``ssa_attribution``.
    # Defaulted/optional, so suites that don't sub-classify and older reports
    # simply leave it null.
    subclass: str | None = None
    # Per-engine timing breakdown (parse/codegen/integrate) and config metadata.
    # Populated by rr_parity ODE runs; None for other suites/methods. Structure:
    # {
    #   "bngsim": {
    #     "parse_sec": 0.008, "codegen_sec": 0.034, "integrate_sec": 0.009,
    #     "config": {"codegen": "MIR JIT", "jacobian": "analytical", ...}
    #   },
    #   "roadrunner": {
    #     "parse_sec": 0.007, "codegen_sec": 0.031, "integrate_sec": 0.007,
    #     "config": {"codegen": "LLVM JIT", ...}
    #   }
    # }
    timing: dict | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> JobResult:
        # Ignore unknown keys so an older report still loads after a field is
        # retired (e.g. the legacy per-engine wall_bn/wall_rr, dropped once the
        # per-phase ``timing`` dict superseded them) — cls(**d) would otherwise
        # raise TypeError on the stale keys.
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------- #
# Golden references
# --------------------------------------------------------------------------- #
@dataclass
class Golden:
    """Per-job reference fingerprint a consumer regenerates through its bridge.

    `checksum` is a byte-identity hash within a pinned (version, platform, seed)
    cell. `fingerprint` is the cross-platform numeric fallback (see
    fingerprint.py). `trajectory` is a suite-relative path to a full reference
    trajectory, present only for the hand-selected representative subset.
    """

    model_id: str
    method: str
    checksum: str
    fingerprint: dict
    trajectory: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Golden:
        return cls(**d)


# --------------------------------------------------------------------------- #
# Compact JSON I/O (pretty _meta, one record per line)
# --------------------------------------------------------------------------- #
def _write_records(path: Path, meta: dict, records: list[dict], key: str) -> None:
    meta_s = json.dumps(meta, indent=2)
    body = ",\n".join(json.dumps(r, separators=(",", ":")) for r in records)
    path.write_text('{\n"_meta": ' + meta_s + f',\n"{key}": [\n' + body + "\n]}\n")


def _read_records(path: Path, key: str) -> tuple[dict, list[dict]]:
    data = json.loads(Path(path).read_text())
    return data.get("_meta", {}), data.get(key, [])


def write_manifest(path: str | Path, jobs: list[Job], meta: dict | None = None) -> None:
    meta = dict(meta or {})
    meta.setdefault("schema_version", SCHEMA_VERSION)
    meta.setdefault("generated", _dt.date.today().isoformat())
    meta["n_jobs"] = len(jobs)
    _write_records(Path(path), meta, [j.to_dict() for j in jobs], "jobs")


def read_manifest(path: str | Path) -> tuple[dict, list[Job]]:
    meta, records = _read_records(Path(path), "jobs")
    return meta, [Job.from_dict(r) for r in records]


def write_report(path: str | Path, results: list[JobResult], meta: dict | None = None) -> None:
    meta = dict(meta or {})
    meta.setdefault("schema_version", SCHEMA_VERSION)
    meta.setdefault("generated", _dt.datetime.now().isoformat(timespec="seconds"))
    meta["n_results"] = len(results)
    _write_records(Path(path), meta, [r.to_dict() for r in results], "results")


def read_report(path: str | Path) -> tuple[dict, list[JobResult]]:
    meta, records = _read_records(Path(path), "results")
    return meta, [JobResult.from_dict(r) for r in records]


def write_golden(path: str | Path, golden: list[Golden], meta: dict | None = None) -> None:
    meta = dict(meta or {})
    meta.setdefault("schema_version", SCHEMA_VERSION)
    meta.setdefault("generated", _dt.date.today().isoformat())
    meta["n_golden"] = len(golden)
    _write_records(Path(path), meta, [g.to_dict() for g in golden], "golden")


def read_golden(path: str | Path) -> tuple[dict, list[Golden]]:
    meta, records = _read_records(Path(path), "golden")
    return meta, [Golden.from_dict(r) for r in records]
