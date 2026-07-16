"""PSA cost/precision decision helper (GH #15).

Partial scaling accelerates a run per path but **inflates the variance** of the
scaled channels. Whether that is a net win at fixed statistical precision depends
on the observable: you must weigh the measured per-path speedup ``ŝ`` (from
:attr:`bngsim.Result.psa_diagnostics`) against the observable's
variance-inflation ratio ``ρ = v_scaled / v_true``.

The decision rule comes from standard sampling statistics (``Var(mean) = v/n``;
``Var(sample variance) ≈ (κ−1)v²/n``); see docs/notes.tex, Proposition
"Fixed-precision cost for mean and variance estimation". At fixed precision,
scaling reduces total sampling cost:

- **for the mean** only when ``ŝ > ρ``  (cost ratio ``ρ/ŝ``);
- **for the (Gaussian) variance** only when ``ŝ > ρ²``  (cost ratio ``ρ²/ŝ``);
  for a non-Gaussian observable the factor ``(κ_s−1)/(κ_0−1)`` multiplies ``ρ²/ŝ``.

``ŝ`` is measurable from the scaled run alone, but ``ρ`` is **not** — it needs a
calibration/anchor run or a supplied excess model. This helper therefore makes
``rho`` a required argument; it will not invent an inflation factor.

Caveats (this is a best-case benchmark, not a guarantee):

- ``ŝ`` here is an event-rate ratio; it ignores per-event overhead, so it
  overstates wall-clock speedup.
- The result assumes the mean-bias correction (if any) is deterministic and
  exact. A noisy excess correction adds variance and worsens the true cost.
- The variance rule's ``ρ²`` is the Gaussian benchmark; pass measured excess
  kurtoses for a non-Gaussian observable.
"""

from __future__ import annotations

from typing import Any, Literal


def psa_cost_decision(
    speedup: float,
    rho: float,
    *,
    kind: Literal["mean", "variance"] = "mean",
    kappa_scaled: float = 3.0,
    kappa_true: float = 3.0,
) -> dict[str, Any]:
    r"""Decide whether PSA is a net win at fixed precision for one observable.

    Parameters
    ----------
    speedup:
        The measured per-path event-rate ratio ``ŝ`` (``> 0``), e.g.
        ``result.psa_diagnostics["speedup"]``.
    rho:
        The variance-inflation ratio ``ρ = v_scaled / v_true`` (``> 0``) for the
        observable of interest. **Required** — obtain it from a calibration/anchor
        run or an excess model; it cannot be read from the scaled run alone.
    kind:
        ``"mean"`` for sample-mean estimation (threshold ``ρ``) or ``"variance"``
        for sample-variance estimation (Gaussian threshold ``ρ²``).
    kappa_scaled, kappa_true:
        Kurtoses ``κ_s = μ4_s/v_s²`` and ``κ_0 = μ4_0/v_0²`` of the scaled and
        true laws, used only for ``kind="variance"``. Both default to ``3``
        (Gaussian), recovering the ``ρ²`` rule. Each must be ``> 1``.

    Returns
    -------
    dict with keys:

    - ``kind`` (str), ``speedup`` (float), ``rho`` (float)
    - ``threshold`` (float): the speedup ``ŝ`` must exceed for a net win.
    - ``cost_ratio`` (float): total-cost ratio ``n_s c_s / (n_0 c_0)`` at matched
      precision; ``< 1`` means scaling is cheaper.
    - ``net_win`` (bool): ``cost_ratio < 1``.
    - ``message`` (str): a human-readable go/no-go summary.
    """
    if not speedup > 0.0:
        raise ValueError(f"speedup must be > 0, got {speedup!r}")
    if not rho > 0.0:
        raise ValueError(f"rho must be > 0, got {rho!r}")
    if kind not in ("mean", "variance"):
        raise ValueError(f"kind must be 'mean' or 'variance', got {kind!r}")

    if kind == "mean":
        threshold = rho
        cost_ratio = rho / speedup
    else:
        if not kappa_scaled > 1.0 or not kappa_true > 1.0:
            raise ValueError(
                "kappa_scaled and kappa_true must be > 1 for kind='variance', "
                f"got {kappa_scaled!r}, {kappa_true!r}"
            )
        factor = (kappa_scaled - 1.0) / (kappa_true - 1.0)
        threshold = factor * rho * rho
        cost_ratio = threshold / speedup

    net_win = cost_ratio < 1.0
    verb = "reduces" if net_win else "does NOT reduce"
    message = (
        f"PSA {verb} fixed-precision {kind} cost: "
        f"ŝ={speedup:.3g} vs threshold {threshold:.3g} "
        f"(ρ={rho:.3g}); cost ratio {cost_ratio:.3g}."
    )
    return {
        "kind": kind,
        "speedup": float(speedup),
        "rho": float(rho),
        "threshold": float(threshold),
        "cost_ratio": float(cost_ratio),
        "net_win": bool(net_win),
        "message": message,
    }
