"""Shared machinery for the parity_checks suites — contracts only, no models.

Every suite (rr_parity, bng_parity, amici_parity) and every downstream consumer
that regenerates our golden references imports its taxonomy, manifest/report
schema, comparison oracles, fingerprint tooling, and version stamping from
here, so reports are directly comparable across suites and across the
engine/bridge layers.
"""

from __future__ import annotations

from . import bngpath, differ, fingerprint, oracles, versions
from .bngpath import BngResolution, require_bng, resolve_bng, skip_reason
from .schema import (
    EXCEPTION_TEXT_LIMIT,
    SCHEMA_VERSION,
    CapturedException,
    Golden,
    Job,
    JobResult,
    Oracle,
    Override,
    capture_exception,
    read_golden,
    read_manifest,
    read_report,
    write_golden,
    write_manifest,
    write_report,
)
from .taxonomy import ALL, CLEAN, FAILING, Outcome, tally

__all__ = [
    "EXCEPTION_TEXT_LIMIT",
    "SCHEMA_VERSION",
    "BngResolution",
    "CapturedException",
    "capture_exception",
    "bngpath",
    "require_bng",
    "resolve_bng",
    "skip_reason",
    "Outcome",
    "ALL",
    "CLEAN",
    "FAILING",
    "tally",
    "Oracle",
    "Override",
    "Job",
    "JobResult",
    "Golden",
    "read_manifest",
    "write_manifest",
    "read_report",
    "write_report",
    "read_golden",
    "write_golden",
    "oracles",
    "differ",
    "fingerprint",
    "versions",
]
