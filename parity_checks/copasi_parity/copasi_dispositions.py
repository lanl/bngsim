"""COPASI-calibrated per-row dispositions for copasi_parity (issue #1550).

copasi_parity honors only the engine-agnostic ``tol`` overrides from the shared
manifest. rr_parity's ``known_artifact`` / ``invalid_reference`` /
``no_oracle_adjudicated`` entries are calibrated against RoadRunner, and AMICI's
against AMICI, so applying either here could mask a real bngsim-vs-COPASI
difference. That leaves a row where **COPASI** is the wrong engine nowhere to go
but a bngsim ``DIFF``. This module is the COPASI-calibrated path, as
``amici_parity/amici_dispositions.py`` is AMICI's (issue #380).

What an entry does to a row is AMICI's code, not a copy of it: a row is resolved
by ``amici_dispositions.apply_disposition``, so the two suites cannot drift on it.
The lookup helpers below mirror amici_dispositions' for the one kind authored
here. In short:

  * INVALID_REFERENCE turns a ``DIFF`` into ``REFERENCE_FAILED`` with the
    ``invalid_result`` refusal class: non-scoring, and not a PASS, because the
    engines did not agree.
  * A row that PASSes on its own is flagged STALE (prune the entry); a row that
    produced no comparison (EXCEPTION, TIMEOUT, ...) leaves the entry
    inapplicable, and the row keeps scoring.
  * An entry records the ``max_rel`` it was written against; a fresh value more
    than a factor of 2 away is still disposed of, but the comment says DRIFTED.

Only INVALID_REFERENCE is authored so far. AMICI's other two kinds
(COMPARISON_ARTIFACT, NO_GROUND_TRUTH) would mean the same here; adding one takes
a dict for it, and reading that dict in ``disposition_for`` and ``ALL_KEYS``, as
amici_dispositions does.

The bar is AMICI's and rr_parity's: an entry that turned a real bngsim defect
non-scoring would hide a regression, so every entry cites evidence from an oracle
that is NOT bngsim (RoadRunner, or the model's own arithmetic), and names which
engine is wrong and how that is known.

Keys are ``"<model_id>:ode"``; copasi_parity runs only the ODE regime.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PC = Path(__file__).resolve().parent.parent
if str(_PC) not in sys.path:  # so `amici_parity` and `_core` resolve standalone
    sys.path.insert(0, str(_PC))
from amici_parity import amici_dispositions as _ad  # noqa: E402

# Re-exported so a caller needs only this module: the class an INVALID_REFERENCE
# row's ``reference_refusal`` carries, and the function that resolves a row.
INVALID_RESULT_REFUSAL = _ad.INVALID_RESULT_REFUSAL
apply_disposition = _ad.apply_disposition

REGIMES = ("ode",)

# The kind string apply_disposition reads for "the reference is wrong". Private
# in amici_dispositions; read from there rather than spelled again, so a rename
# fails at import instead of silently resolving no row.
_KIND_INVALID_REFERENCE = _ad._KIND_INVALID_REFERENCE


# --------------------------------------------------------------------------- #
# COPASI ran, and its answer is not a usable oracle -> DIFF becomes
# REFERENCE_FAILED / invalid_result. Keyed "model_id:ode".
# --------------------------------------------------------------------------- #
INVALID_REFERENCE: dict[str, dict] = {
    # BIOMD0000000183:ode. PASS at 0.15.1 with 24873 of 67067 cells failing and
    # all of them forgiven; the differ before #873 held a column to the 5%
    # ceiling at its own peak and nothing else. #873 holds every moving species
    # to its own range, which this row fails. bngsim's trajectory
    # did not move: the v0.15.1 differ still passes it against COPASI, and
    # RoadRunner fails against COPASI exactly as bngsim does.
    "BIOMD0000000183:ode": {
        "issue": "GH #1550",
        "max_rel": 0.6197,
        "reason": (
            "COPASI's SBML import rewrites four reactions as mass action and drops "
            "a factor of their rate. reaction_129 to reaction_132 read "
            "compartment_0 * transition1_R_T(kRT, b, camR_ca1_X), with "
            "transition1_R_T(ka, b, species) = species * ka * b^(1/2), where b is "
            "kTR (reaction_129) or cB, cC, cD. COPASI 4.46.300 imports each as "
            "'Mass action (irreversible)' with k1 mapped to kRT, so its rate lacks "
            "b^(1/2). With RoadRunner's t = 0.01 state written into the SBML, "
            "COPASI's rates for those four are RoadRunner's divided by sqrt(b) "
            "(6.955 = sqrt(48.38) on reaction_129, 15.89 = 1/sqrt(0.00396) on "
            "reaction_131), and the other 348 agree to 1e-15. The same holds on a "
            "one-reaction model, rate c * S * ka * bb^(1/2) = 0.5: COPASI reads 1.0 "
            "whether the power is written inline, through a function definition, "
            "or as a root, and RoadRunner reads 0.5; without the power, COPASI "
            "reads it right. COPASI's trajectory is converged to that other model: "
            "the same 24599 failing cells and max_rel 0.620 at rtol/atol 1e-9/1e-12, "
            "1e-10/1e-14 and 1e-12/1e-20, while bngsim and RoadRunner converge to one answer "
            "(0 of 67067 cells failing at the suite tolerance; at 1e-12/1e-20 the "
            "two agree to 1.4e-10 of each species' range). "
            "RoadRunner against COPASI fails 24599 cells, the same as bngsim."
        ),
    },
    # BIOMD0000000285:ode. PASS at 0.15.1, when bngsim did not fire these two
    # events either (#1150 changed that, by the decision on #934), and the differ
    # before #873 forgave the disagreement anyway: it passes today's firing bngsim
    # against COPASI too (10759 cells failing, 0 hard). AMICI reads the model as
    # COPASI does; amici_dispositions carries the AMICI row (#1546).
    "BIOMD0000000285:ode": {
        "issue": "GH #1550",
        "max_rel": 1.0,
        "reason": (
            "COPASI does not fire a trigger whose residual sits at exactly 0 until "
            "it turns true, unless it leaves 0 with a nonzero slope at t = 0. The "
            "model's two death events, PIcellDeath (PIdeath > 0) and p38cellDeath "
            "(p38death > 0), each set kalive = 0, which stops every reaction. "
            "PIdeath and p38death start at 0 with a rate of exactly 0 and are "
            "positive from just after t = 0. COPASI fires neither, and the cell "
            "lives (NatP 19520.1, MisP 3.899 at t = 10, AMICI's numbers too, "
            "#1546). bngsim (by the decision on #934, #1150) and RoadRunner fire "
            "both just after t = 0 (NatP 19500) and agree on all 27027 cells; "
            "RoadRunner against COPASI fails the same 11463 cells (9909 hard) as "
            "bngsim. COPASI's answer is not the other reading of the start rule: "
            "written as L3V2 with initialValue = false, where both triggers are "
            "false before t = 0 and true at every t > 0, so that every reading "
            "fires them, COPASI still fires neither (NatP 19520.1), and RoadRunner "
            "and bngsim fire both. On B > 0 with B(0) = 0 COPASI fires "
            "dB/dt = 1 and dB/dt = 1e-14 at t = 0 with either initialValue, as "
            "RoadRunner and bngsim do just after it, but never fires dB/dt = t, "
            "nor B held at 0 until t = 0.5 and rising after, a crossing every "
            "reading fires; RoadRunner and bngsim fire both, just after 0 and at "
            "0.5. None of the three fires B > 0 with B falling from 0 or held at "
            "0, nor B < 0 with B rising. So COPASI is not an oracle at a trigger "
            "that sits on its threshold, whichever reading of the start rule is "
            "right."
        ),
    },
}

ALL_KEYS = frozenset(INVALID_REFERENCE)


def disposition_for(model_id: str, regime: str) -> dict | None:
    """The authored disposition for ``(model_id, regime)``, or None, in the shape
    ``apply_disposition`` reads: ``{"kind", "reason", "issue", "max_rel"}``."""
    d = INVALID_REFERENCE.get(f"{model_id}:{regime}")
    if d is None:
        return None
    return {
        "kind": _KIND_INVALID_REFERENCE,
        "reason": d["reason"],
        "issue": d.get("issue"),
        "max_rel": d.get("max_rel"),
    }


def dispositions_for_regime(regime: str) -> dict[str, dict]:
    """Every authored disposition for ``regime``, keyed by bare ``model_id``, the
    shape the runner wants. Built once per run rather than per row."""
    out: dict[str, dict] = {}
    for key in ALL_KEYS:
        model_id, _, key_regime = key.rpartition(":")
        if key_regime == regime:
            out[model_id] = disposition_for(model_id, regime)  # type: ignore[assignment]
    return out


def stale_keys(model_ids: set[str], regime: str) -> list[str]:
    """Authored keys for ``regime`` whose model is NOT in ``model_ids``.

    Only meaningful against the full model set, which is why the runner does not
    call it (a ``--models`` run omits most of them) and the test suite does.
    """
    return sorted(
        k for k in ALL_KEYS if k.endswith(f":{regime}") and k.rsplit(":", 1)[0] not in model_ids
    )
