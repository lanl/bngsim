"""SSA-compatibility validation for SBML-loaded models.

The SBML loader (``_sbml_loader.py``) records SSA-relevant constructs at
load time and stashes the resulting :class:`SsaIssue` records on
``Model._ssa_issues``. :func:`validate_for_ssa` returns that list, plus
runtime warnings for fractional initial SSA populations that the simulator
will round at setup.

:class:`bngsim.Simulator` calls :func:`validate_for_ssa` when
``method="ssa"`` is requested and raises
:class:`bngsim.SsaValidationError` if any issue has
``severity == "error"``. Warnings are emitted via the ``bngsim`` logger.

See ``dev/plans/SBML_SSA_SUPPORT_PLAN.md`` Phase 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from bngsim._model import Model


@dataclass(frozen=True)
class SsaIssue:
    """A single SSA-compatibility finding from an SBML model.

    Attributes
    ----------
    severity
        ``"error"`` (Simulator init will raise) or ``"warning"`` (logged).
    code
        Stable machine-readable identifier for the issue category.
    message
        Human-readable description, ideally pointing the user at a fix.
    location
        Optional locator string (e.g. ``"reaction:R1"``,
        ``"compartment:cell"``, ``"species:S"``).
    """

    severity: Literal["error", "warning"]
    code: str
    message: str
    location: str | None = None


def validate_for_ssa(model: Model) -> list[SsaIssue]:
    """Return SSA-compatibility issues recorded for *model*.

    The list is populated by the SBML loader at load time, with an
    additional model-agnostic warning when the current species state
    contains non-integer molecule populations. Fractional populations are
    rounded by the C++ SSA setup path before simulation.

    PyBNF and other consumers can call this before constructing a
    ``Simulator`` to pre-flight a model — the same list will be re-checked
    by ``Simulator(model, method="ssa")``.
    """
    issues = list(getattr(model, "_ssa_issues", None) or [])

    try:
        species_meta = list(model._core.codegen_data()["species"])
    except Exception:
        species_meta = []

    for sp in species_meta:
        name = str(sp.get("name", ""))
        volume_factor = float(sp.get("volume_factor", 1.0))
        if not name or not math.isfinite(volume_factor) or volume_factor <= 0.0:
            continue
        try:
            storage_value = float(model.get_concentration(name))
        except Exception:
            continue
        amount = storage_value * volume_factor
        if not math.isfinite(amount):
            continue
        rounded_amount = math.floor(amount + 0.5) if amount >= 0.0 else math.ceil(amount - 0.5)
        if abs(amount - rounded_amount) <= 1e-9 * max(1.0, abs(amount)):
            continue
        issues.append(
            SsaIssue(
                severity="warning",
                code="non_integer_initial_population",
                message=(
                    "Initial SSA population is non-integer; bngsim will round it "
                    f"to {rounded_amount:g} before simulation."
                ),
                location=f"species:{name}",
            )
        )

    return issues
