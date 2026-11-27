"""Frozen references: enclosures recorded offline by validated integrators
(issue #1912).

CAPD and Ariadne compute guaranteed enclosures: bands that hold the true
solution, and certified intervals that hold each crossing time. Both are
GPL-3.0 C++ libraries, so neither may be a dependency of bngsim or of CI.
They run offline, as recorders, and what they find is committed here as data
and read back by ``Frozen``.

Each tool has its own job (#1781's plan; the recorders land with the first
cases that need them, #1915):

* Ariadne certifies the crossing structure: which crossings occur, in what
  order, and an interval for each time. Guards, resets and modes are its
  native model.
* CAPD encloses the sensitivity columns; its C1 solvers bound the first-order
  variational equations along each branch.
* Both enclose the species at the recorded times, and their bands must
  overlap. Bands that do not overlap mean a bug in one tool or a case written
  wrongly into one tool's input; the record then refuses to answer.

The record (JSON), one per case::

    {"schema": 1, "case": "<name>", "case_fingerprint": "<sha256>",
     "rows": [{"params": {...}, "observe": [...], "sens": [...],
               "times": [...],
               "tools": {"<tool>": {"version": "<pin>",
                                    "species": {"lo": [[...]], "hi": [[...]]},
                                    "sens": {"lo": [[[...]]], "hi": [[[...]]]} | null,
                                    "crossings": [{"switch": "...", "lo": t, "hi": t}] | null}}}]}

``species`` is (n_times, n_observe), ``sens`` (n_times, n_observe, n_sens).
A record is tied to its case by a fingerprint of the model text, the window,
the observed species, the base values and the spec; one recorded for another
version of the case is stale and refuses to answer, as the nightly's
reference archive does.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from _reference import Band, ReferenceUnavailable, Solution

SCHEMA = 1
TOOLS = ("ariadne", "capd")


def case_fingerprint(case) -> str:
    """What a record of this case depends on, hashed."""
    parts = {
        "model": case.model,
        "fmt": case.fmt,
        "t_span": list(case.window()),
        "observe": list(case.observe),
        "params": {k: float(v) for k, v in sorted(case.params.items())},
    }
    if case.spec is not None:
        sym = case.spec.symbolic()
        parts["spec"] = {
            "states": list(case.spec.states),
            "params": list(case.spec.params),
            "f": [str(e) for e in sym.f],
            "g": [str(e) for e in sym.g],
            "x0": [str(e) for e in sym.x0],
        }
    blob = json.dumps(parts, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _params_key(params: Mapping[str, float], names: Sequence[str]) -> tuple:
    return tuple((q, float(params[q])) for q in sorted(names))


@dataclass(eq=False)
class Frozen:
    """A committed record of enclosures for one case."""

    path: Path
    kind: str = field(default="frozen", init=False)
    _record: dict | None = field(default=None, init=False, repr=False)
    _fingerprint: str | None = field(default=None, init=False, repr=False)

    def check_case(self, case) -> None:
        self._fingerprint = case_fingerprint(case)

    def record(self) -> dict:
        if self._record is None:
            try:
                rec = json.loads(Path(self.path).read_text())
            except FileNotFoundError:
                raise ReferenceUnavailable(f"no record at {self.path}") from None
            if rec.get("schema") != SCHEMA:
                raise ReferenceUnavailable(
                    f"{self.path}: schema {rec.get('schema')}, not {SCHEMA}"
                )
            self._record = rec
        rec = self._record
        if self._fingerprint is None:
            raise ReferenceUnavailable("a Frozen reference is read only through its case")
        if rec.get("case_fingerprint") != self._fingerprint:
            raise ReferenceUnavailable(
                f"{self.path} is stale: it was recorded for another version of case "
                f"{rec.get('case')!r}; re-record it"
            )
        return rec

    def _row(self, params: Mapping[str, float]) -> dict:
        """The row recorded at exactly these values: every parameter the run
        sets, no more and no fewer."""
        rec = self.record()
        want = _params_key(params, list(params))
        for row in rec["rows"]:
            if _params_key(row["params"], list(row["params"])) == want:
                return row
        raise ReferenceUnavailable(f"{self.path} holds no record for these parameters")

    def crossings(self, params: Mapping[str, float], t_span) -> tuple[float, ...]:
        return tuple(0.5 * (lo + hi) for lo, hi, _ in self.intervals(params))

    def intervals(self, params: Mapping[str, float]) -> list[tuple[float, float, str]]:
        """Each crossing's certified interval, the tools' intervals intersected."""
        row = self._row(params)
        given = {
            t: s["crossings"] for t, s in row["tools"].items() if s.get("crossings") is not None
        }
        if not given:
            return []
        counts = {t: len(c) for t, c in given.items()}
        if len(set(counts.values())) != 1:
            raise ReferenceUnavailable(f"the tools find different numbers of crossings: {counts}")
        out = []
        for i in range(next(iter(counts.values()))):
            each = [c[i] for c in given.values()]
            if len({e["switch"] for e in each}) != 1:
                raise ReferenceUnavailable(f"the tools order crossing {i} differently: {each}")
            lo, hi = max(e["lo"] for e in each), min(e["hi"] for e in each)
            if lo > hi:
                raise ReferenceUnavailable(
                    f"the tools' intervals for crossing {i} do not overlap: {each}"
                )
            out.append((lo, hi, each[0]["switch"]))
        return out

    def certify(self, params: Mapping[str, float], crossings: Sequence[tuple[float, str]]) -> None:
        """Raise unless ``crossings`` (time, switch) match the certified
        structure: the same switches in the same order, each time inside its
        interval."""
        want = self.intervals(params)
        if len(want) != len(crossings):
            raise ReferenceUnavailable(
                f"{len(crossings)} crossings where the record certifies {len(want)}"
            )
        for (t, sw), (lo, hi, wsw) in zip(crossings, want, strict=True):
            if sw != wsw or not (lo <= t <= hi):
                raise ReferenceUnavailable(
                    f"crossing of {sw!r} at t = {t!r} is not the certified one: {wsw!r} in [{lo!r}, {hi!r}]"
                )

    def solve(
        self, params, t_span, times, observe: Sequence[str], sens: Sequence[str]
    ) -> Solution:
        row = self._row(params)
        t = np.asarray(times, dtype=float)
        at = {float(x): i for i, x in enumerate(row["times"])}
        missing = [float(x) for x in t if float(x) not in at]
        if missing:
            raise ReferenceUnavailable(f"no enclosure recorded at t = {missing[:3]}")
        rows = [at[float(x)] for x in t]
        o_idx = _index(row["observe"], observe, "species")
        s_idx = _index(row["sens"], sens, "column") if sens else []
        species = _intersect(row, "species", rows, o_idx, None)
        sens_band = _intersect(row, "sens", rows, o_idx, s_idx) if sens else None
        cross = self.crossings(params, t_span)
        return Solution(
            times=t,
            species=species,
            species_left=species,
            sens=sens_band,
            sens_left=sens_band,
            crossings=cross,
            at_crossing=np.zeros(t.shape, dtype=bool),
            floor=0.0,
            kind=self.kind,
            note="enclosure from " + "+".join(sorted(row["tools"])),
        )


def _index(have: Sequence[str], want: Sequence[str], what: str) -> list[int]:
    missing = [w for w in want if w not in have]
    if missing:
        raise ReferenceUnavailable(f"the record encloses no {what} {missing}")
    return [list(have).index(w) for w in want]


def _intersect(row: dict, key: str, rows, o_idx, s_idx) -> Band:
    bands = []
    for tool, sec in sorted(row["tools"].items()):
        if sec.get(key) is None:
            continue
        lo, hi = np.asarray(sec[key]["lo"], float), np.asarray(sec[key]["hi"], float)
        lo, hi = lo[rows][:, o_idx], hi[rows][:, o_idx]
        if s_idx is not None:
            lo, hi = lo[:, :, s_idx], hi[:, :, s_idx]
        bands.append((tool, lo, hi))
    if not bands:
        raise ReferenceUnavailable(f"no tool in the record encloses the {key}")
    lo = np.max([b[1] for b in bands], axis=0)
    hi = np.min([b[2] for b in bands], axis=0)
    if np.any(lo > hi):
        bad = np.argwhere(lo > hi)[0]
        raise ReferenceUnavailable(
            f"the {key} enclosures of {', '.join(b[0] for b in bands)} do not overlap at index "
            f"{tuple(int(i) for i in bad)}: a bug in one tool, or the case written wrongly into one"
        )
    return Band(lo, hi)


def tool_section(
    version: str,
    species: Band,
    sens: Band | None = None,
    crossings: Sequence[tuple[str, float, float]] | None = None,
) -> dict:
    """One tool's part of a record row."""
    return {
        "version": version,
        "species": {"lo": species.lo.tolist(), "hi": species.hi.tolist()},
        "sens": None if sens is None else {"lo": sens.lo.tolist(), "hi": sens.hi.tolist()},
        "crossings": None
        if crossings is None
        else [{"switch": s, "lo": float(lo), "hi": float(hi)} for s, lo, hi in crossings],
    }


def write_record(path: Path, case, rows: Sequence[dict]) -> None:
    """Write a case's record. Each row: ``params``, ``observe``, ``sens``,
    ``times`` and ``tools`` (tool name -> ``tool_section``)."""
    for r in rows:
        stray = set(r["tools"]) - set(TOOLS)
        if stray:
            raise ValueError(f"unknown tools {sorted(stray)}; the record knows {TOOLS}")
    rec = {
        "schema": SCHEMA,
        "case": case.name,
        "case_fingerprint": case_fingerprint(case),
        "rows": [
            {
                "params": {k: float(v) for k, v in r["params"].items()},
                "observe": list(r["observe"]),
                "sens": list(r["sens"]),
                "times": [float(t) for t in r["times"]],
                "tools": r["tools"],
            }
            for r in rows
        ],
    }
    Path(path).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
