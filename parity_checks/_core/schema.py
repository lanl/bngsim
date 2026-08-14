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
# Exception capture for a report row (issue #324)
# --------------------------------------------------------------------------- #
# A report row's `exception` is a bounded field: 2,646 rows carrying an unbounded
# traceback repr would make the durable artifact unreadable and unwieldy, so the
# text is capped. The cap used to be a plain head cut, and that is what #324
# found: several bngsim refusals put the phrase naming the failure CLASS at the
# END, after an enumeration of model symbols. The under-specified-model refusal
# (#323) reads "Parameters 'A', 'B', ... have no value attribute and no
# initialAssignment, but are referenced by ..." — on a model with a long
# parameter list all 400 characters were names and the diagnostic never appeared.
# Three models could not be classified from the report at all and had to be
# grouped by hand against their source, on a sweep that costs ~4 hours to redo.
EXCEPTION_TEXT_LIMIT = 400

# Head/tail split of the surviving budget. The head carries the phase, exception
# type and opening clause; the tail carries the trailing diagnostic an
# enumeration would otherwise push out.
_ELIDE_HEAD_NUM, _ELIDE_HEAD_DEN = 3, 5
_ELIDE_MARKER = " ...[{n} chars elided]... "


def _elide_middle(text: str, limit: int) -> str:
    """``text`` capped at ``limit`` chars, dropping the MIDDLE rather than the tail.

    The marker names how many characters went, so an elided message can never be
    mistaken for one that genuinely reads that way. The result is exactly
    ``limit`` characters when anything was dropped, so the cap on report size is
    the same one the head cut gave.

    The marker's own width depends on the digit count of the number it prints,
    which depends on how much the marker leaves room to keep — so the split is a
    two-step fixpoint rather than one subtraction.
    """
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    marker = _ELIDE_MARKER.format(n=dropped)
    for _ in range(4):
        keep = limit - len(marker)
        if keep < 2:  # pathologically small limit — nothing sane to split
            return text[:limit]
        if len(text) - keep == dropped:
            break
        dropped = len(text) - keep
        marker = _ELIDE_MARKER.format(n=dropped)
    keep = limit - len(marker)
    head = keep * _ELIDE_HEAD_NUM // _ELIDE_HEAD_DEN
    tail = keep - head
    return f"{text[:head]}{marker}{text[len(text) - tail :]}"


@dataclass(frozen=True)
class CapturedException:
    """One exception, rendered for a report row three ways (issue #324).

    ``full`` is the untruncated text and never reaches the report — it exists so
    a *classifier* runs against the whole message rather than against whatever
    survived the cap. That distinction is the point: a keyword sitting past the
    head budget used to decide a subclass only by accident of message length.
    """

    text: str  # "<phase>: <Type>: <message>", middle-elided to the limit
    cls: str  # "<phase>:<Type>" — the stable grouping key
    full: str  # the same text, uncapped; for classifiers, not for the report


def capture_exception(
    phase: str, exc: BaseException, limit: int = EXCEPTION_TEXT_LIMIT
) -> CapturedException:
    """Render ``exc`` for a report row, tagged with the ``phase`` that raised it.

    ``phase`` is the step in the job the raise came from — ``"bngsim"``,
    ``"amici"``, ``"amici-build"``, ``"bngsim-params"``, ``"compare"``. Paired
    with the exception type it gives ``cls``, a key that is **stable across
    models**: two models failing the same way group together however different
    their symbol names are, which is exactly what a report-only census needs and
    what a message truncated mid-enumeration cannot give.

    Deliberately not a parsed subclass. A leading clause is a guess about message
    wording; the type is a fact, and it is already the axis
    ``is_declared_refusal`` matches on.
    """
    name = type(exc).__name__
    full = f"{phase}: {name}: {exc}"
    return CapturedException(text=_elide_middle(full, limit), cls=f"{phase}:{name}", full=full)


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
    # Stable grouping key for `exception` — "<phase>:<ExceptionType>", or two of
    # those joined by " || " when both engines raised (issue #324). `exception`
    # is capped and its variable part (a symbol enumeration) can be arbitrarily
    # long, so it is not a key a census can group on; this is. Defaulted/optional,
    # so suites that don't set it and older reports simply leave it null.
    exception_class: str | None = None
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
    # Free-form per-row facts a specific regime needs to carry that are neither a
    # verdict, a timing, nor a sub-classification — kept out of ``comment`` so a
    # renderer can read them as data instead of parsing prose. Example: the
    # amici_parity forward-sensitivity job records ``{"sens_method": "staggered",
    # "n_params": 20, "n_param_candidates": 43, "n_common_species": 8,
    # "state_passed": true, "state_max_rel": 3.1e-09}`` — the parameter count is
    # required to interpret that row's timing at all (cost scales with Np), and the
    # state verdict qualifies the sensitivity verdict. Defaulted/optional, so every
    # other suite and every older report simply leaves it null.
    extra: dict | None = None

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
