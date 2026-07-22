"""The PyBioNetGen pin exists in two files; this fails when they drift.

``parity_checks/requirements-pybionetgen.txt`` is the source of truth — it
carries the rationale for the pin and is what ``bootstrap_parity_env.py``
installs. ``pyproject.toml``'s ``[dependency-groups] parity`` repeats the same
commit so ``uv sync --group parity`` provisions the identical bridge.

Two copies of a sha is a drift hazard: bump one (as GH #4 bumped the
requirements file) and the other silently keeps installing the old commit, so
`uv sync` and `bootstrap_parity_env.py` would provision *different* upstreams
and disagree about routing — exactly the class of "which code did this actually
run?" problem the pin exists to prevent. Same tripwire pattern as the vendored
NFsim carry-queue check.
"""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

_PARITY = Path(__file__).resolve().parent.parent
_REPO = _PARITY.parent
_REQ = _PARITY / "requirements-pybionetgen.txt"
_PYPROJECT = _REPO / "pyproject.toml"

_PIN_RE = re.compile(
    r"bionetgen\s*@\s*git\+https://github\.com/RuleWorld/PyBioNetGen\.git@(?P<sha>[0-9a-f]+)"
)


def _sha(text: str, where: str) -> str:
    matches = _PIN_RE.findall(text)
    assert matches, f"no RuleWorld/PyBioNetGen pin found in {where}"
    assert len(set(matches)) == 1, (
        f"{where} pins several different commits: {sorted(set(matches))}"
    )
    return matches[0]


def test_requirements_and_dependency_group_pin_the_same_commit():
    req_sha = _sha(_REQ.read_text(), _REQ.name)

    groups = tomllib.loads(_PYPROJECT.read_text())["dependency-groups"]
    parity = groups["parity"]
    group_sha = _sha("\n".join(parity), "pyproject.toml [dependency-groups] parity")

    # Either may be the shorter prefix of the other; compare on the common length
    # so a 7-char pin and a full 40-char sha of the same commit still agree.
    n = min(len(req_sha), len(group_sha))
    assert req_sha[:n] == group_sha[:n], (
        f"PyBioNetGen pin drift: {_REQ.name} pins {req_sha!r} but pyproject.toml's "
        f"`parity` group pins {group_sha!r}. requirements-pybionetgen.txt is the "
        "source of truth — update the group to match it."
    )


def test_uv_disables_build_isolation_for_bionetgen():
    """Without this, `uv sync --group parity` cannot build PyBioNetGen.

    Its setup.py shells out to `pip install numpy` and downloads BNG2.pl at build
    time, so an isolated build env fails. Dropping the setting would make the
    documented one-command provisioning path break.
    """
    cfg = tomllib.loads(_PYPROJECT.read_text()).get("tool", {}).get("uv", {})
    assert "bionetgen" in cfg.get("no-build-isolation-package", []), (
        "pyproject.toml [tool.uv] must list bionetgen in no-build-isolation-package"
    )
