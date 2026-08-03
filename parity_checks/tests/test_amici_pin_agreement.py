"""The AMICI pin exists in two files; this fails when they drift.

``parity_checks/amici_parity/AMICI_PIN.json`` is the source of truth — it
carries the rationale, the `describe` string, and is what every AMICI verdict in
``AMICI_KNOWN_ISSUES.md`` is tied to. ``pyproject.toml``'s
``[dependency-groups] amici`` repeats the same commit so ``uv sync --group
amici`` provisions the identical reference engine.

Same tripwire as :mod:`test_pin_agreement` does for PyBioNetGen, and the reason
is sharper here: an unpinned or drifted reference engine makes a cross-engine
disagreement unattributable. AMICI_PIN.json exists precisely because the
previous state — an editable build from a developer's home directory, recorded
nowhere — left every recorded PASS/DIFF/timing tied to an environment nobody
could rebuild.

The pin also may not be relaxed to the PyPI release: it is 12 commits past
``v1.0.1``, and those commits touch ``_symbolic/de_model.py``, the exporters and
the SWIG template — the engine, not just its test data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import tomllib

_PARITY = Path(__file__).resolve().parent.parent
_REPO = _PARITY.parent
_PIN = _PARITY / "amici_parity" / "AMICI_PIN.json"
_PYPROJECT = _REPO / "pyproject.toml"

_PIN_RE = re.compile(
    r"amici\s*@\s*git\+https://github\.com/AMICI-dev/AMICI\.git@(?P<sha>[0-9a-f]+)"
)


def test_pin_json_and_dependency_group_pin_the_same_commit():
    pin_sha = json.loads(_PIN.read_text())["commit"]

    groups = tomllib.loads(_PYPROJECT.read_text())["dependency-groups"]
    matches = _PIN_RE.findall("\n".join(groups["amici"]))
    assert matches, "no AMICI-dev/AMICI pin found in pyproject.toml's `amici` group"
    assert len(set(matches)) == 1, f"the group pins several commits: {sorted(set(matches))}"
    group_sha = matches[0]

    n = min(len(pin_sha), len(group_sha))
    assert pin_sha[:n] == group_sha[:n], (
        f"AMICI pin drift: AMICI_PIN.json pins {pin_sha!r} but pyproject.toml's "
        f"`amici` group pins {group_sha!r}. AMICI_PIN.json is the source of truth — "
        "update the group to match it."
    )


def test_amici_group_is_not_an_extra():
    """A direct reference in ``[project.optional-dependencies]`` makes the
    distribution unpublishable — PyPI rejects any metadata containing one. The
    group is what keeps ``uv sync --group amici`` available without that cost,
    the same trade `parity` makes."""
    data = tomllib.loads(_PYPROJECT.read_text())
    for name, reqs in data.get("project", {}).get("optional-dependencies", {}).items():
        for req in reqs:
            assert "git+" not in req, (
                f"extra {name!r} carries a direct reference ({req!r}); move it to "
                "[dependency-groups] or bngsim cannot be published to PyPI"
            )
