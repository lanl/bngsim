"""Every committed corpus provenance manifest conforms to manifest.schema.json.

The schema (``parity_checks/manifest.schema.json``, JSON Schema draft 2020-12) is
the contract for corpus reproducibility: one receipt per model with a pinned
upstream origin, the sha256 of the tested bytes, and a license. This test locks
that contract for all four committed manifests (bng_parity, rr_parity SED-ML,
biomodels, DSMTS), so a bad vendor/fetch run — an unpinned ``@main`` origin, a
missing license, a patched-without-repairs record — fails loudly instead of
silently shipping a non-reproducible manifest.

Bespoke tool-pins (``SUITE_PIN.json`` / ``AMICI_PIN.json``) are NOT corpus
manifests and deliberately do not use this schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

_PC = Path(__file__).resolve().parent.parent  # parity_checks/
_REPO = _PC.parent  # bngsim/
_SCHEMA = json.loads((_PC / "manifest.schema.json").read_text())

MANIFESTS = {
    "bng_parity": _PC / "bng_parity" / "manifest.json",
    "rr_parity_sedml": _PC / "rr_parity" / "temp_biomodels_manifest.json",
    "biomodels": _REPO / "benchmarks" / "suites" / "biomodels" / "biomodels_provenance.json",
    "dsmts": _REPO / "harness" / "sbml_test_suite" / "dsmts" / "dsmts_manifest.json",
}


def _load(name: str) -> dict:
    return json.loads(MANIFESTS[name].read_text())


@pytest.mark.parametrize("name", sorted(MANIFESTS))
def test_manifest_conforms_to_schema(name):
    validator = jsonschema.Draft202012Validator(_SCHEMA)
    errors = sorted(validator.iter_errors(_load(name)), key=lambda e: list(e.path))
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors[:5])


@pytest.mark.parametrize("name", sorted(MANIFESTS))
def test_record_ids_are_unique(name):
    records = _load(name)["records"]
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), f"{name} has duplicate record ids"


@pytest.mark.parametrize("name", sorted(MANIFESTS))
def test_patched_iff_repairs(name):
    """The schema's documented invariant: patched is true exactly when repairs is non-empty."""
    for r in _load(name)["records"]:
        if "patched" in r:
            assert r["patched"] == bool(r.get("repairs")), (
                f"{name}:{r['id']} patched/repairs mismatch"
            )


def test_bng_parity_licenses_match_source_repos():
    """bng_parity's license follows the origin repo: RuleHub/RuleMonkey MIT, BNGL-Models CC-BY-4.0."""
    repo_spdx = {
        "RuleWorld/RuleHub": "MIT",
        "richardposner/RuleMonkey": "MIT",
        "wshlavacek/BNGL-Models": "CC-BY-4.0",
    }
    for r in _load("bng_parity")["records"]:
        assert r["license"]["spdx"] == repo_spdx[r["origin"]["repo"]], (
            f"{r['id']}: license {r['license']['spdx']} != {repo_spdx[r['origin']['repo']]}"
        )
