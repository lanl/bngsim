"""Golden-reference fingerprint tooling.

Two reductions of a simulation result, both small enough to commit for every
job (full trajectories are kept only for a representative subset):

  checksum     — a sha256 over the trajectory rounded to a fixed number of
                 significant digits. Byte-identity WITHIN a pinned (version,
                 platform, seed) cell: a consumer regenerating through its own
                 bridge on the same pinned bngsim should reproduce it exactly.
  fingerprint  — a compact numeric summary (per-variable final value + min/max/
                 mean/last). The cross-platform fallback: when a checksum
                 mismatches (different BLAS, rounding), the fingerprint is
                 compared with a tolerance instead.

Both operate on a (time, values, names) triple where `values` is shape
(n_time, n_var) and `names` labels the columns.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

# Significant digits the checksum rounds to. Tight enough to catch a real
# divergence, loose enough to survive same-platform re-runs.
CHECKSUM_SIGFIGS = 9


def _round_sig(x: np.ndarray, sig: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.floor(np.log10(np.abs(x)))
        factor = 10.0 ** (sig - 1 - mag)
        out = np.round(x * factor) / factor
    out[x == 0] = 0.0
    out[~np.isfinite(x)] = np.nan
    return out


def checksum(time: np.ndarray, values: np.ndarray, names: list[str]) -> str:
    """sha256 of the sig-fig-rounded (names, time, values) — a byte-identity key."""
    t = _round_sig(np.asarray(time, float), CHECKSUM_SIGFIGS)
    v = _round_sig(np.asarray(values, float), CHECKSUM_SIGFIGS)
    h = hashlib.sha256()
    h.update(("\x1f".join(names)).encode())
    h.update(b"\x1e")
    h.update(np.ascontiguousarray(t).tobytes())
    h.update(b"\x1e")
    h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def fingerprint(time: np.ndarray, values: np.ndarray, names: list[str]) -> dict:
    """Per-variable numeric summary: the cross-platform tolerance fallback."""
    v = np.asarray(values, float)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    summary = {}
    for j, name in enumerate(names):
        col = v[:, j]
        finite = col[np.isfinite(col)]
        summary[name] = {
            "last": float(col[-1]) if col.size else float("nan"),
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "mean": float(finite.mean()) if finite.size else float("nan"),
        }
    return {
        "n_time": int(np.asarray(time).shape[0]),
        "n_var": len(names),
        "t_last": float(np.asarray(time, float)[-1]) if np.asarray(time).size else float("nan"),
        "vars": summary,
    }


def fingerprint_max_rel(a: dict, b: dict, abs_floor: float = 1e-9) -> float:
    """Worst relative difference between two fingerprints' per-variable stats.

    Used to judge a golden regeneration when checksums differ across platforms.
    Compares only the variables present in both; a shape/var-set mismatch is a
    structural difference the caller should treat as a hard fail (returns inf).
    """
    av, bv = a.get("vars", {}), b.get("vars", {})
    if set(av) != set(bv) or a.get("n_time") != b.get("n_time"):
        return float("inf")
    worst = 0.0
    for name in av:
        for stat in ("last", "min", "max", "mean"):
            x, y = av[name][stat], bv[name][stat]
            denom = max(abs(y), abs_floor)
            worst = max(worst, abs(x - y) / denom)
    return worst


def golden_pair(time, values, names) -> tuple[str, dict]:
    """Convenience: (checksum, fingerprint) for one result."""
    return checksum(time, values, names), fingerprint(time, values, names)


def dumps(obj: dict) -> str:
    """Stable JSON for a fingerprint (sorted keys) — handy for debugging diffs."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
