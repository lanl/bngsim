"""GH #181 — a .net's parameter kinds come from the expressions, not the comments.

BNG2.pl closes every parameter line with a kind annotation, ``# Constant`` or
``# ConstantExpression``. Those are *comments*. The codegen .net parser used to
read them as the definition of which parameters are derived, so a hand-written or
third-party ``.net`` that omits them had every parameter taken for a leaf:
``a  p*c1`` contributed no ``∂a/∂p`` to the sensitivity RHS and ``dX/dp`` came
back **identically zero** — no warning, no error, and a Python-visible model that
had classified ``a`` as derived the whole time. A zero column reads to a gradient
fit as "this parameter does not matter", which is the failure mode that does not
announce itself.

The rule that replaced it: a parameter is derived exactly when its value
expression references another declared parameter. Two properties are pinned here.

* **It answers the question.** Stripping the annotations from a ``.net`` changes
  neither the emitted C nor the sensitivities it computes.
* **It does not overreach.** ``pi = 2*asin(1)`` names nothing else, so BNG2.pl
  annotates it ``# Constant`` and this rule agrees: it stays a differentiation
  leaf. Over the 1,817 ``.net`` files in this tree — 41,433 annotated parameter
  lines — the two agree everywhere BNG2.pl has an opinion, which is why no
  annotated model's emitted C moved. Asking instead whether the value *text
  parses as a float* — the other reading of "derive it from the expression" —
  reclassifies 628 of those lines and rewrites the sensitivity RHS of 54 models
  that were never broken.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._codegen import _parse_net_file, generate_sens_rhs_c

_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"

# The issue's model. X is fed at rate a = p*c1 from a clamped source and decays at
# b = p*c2, so dX/dt = p*c1 − p*c2*X and dX/dp = c1*t*exp(−p*c2*t) in closed form.
# Both derived parameters carry the SAME primary, which is what a missing chain
# rule silences completely rather than partially.
P, C1, C2 = 1.0, 2e18, 1.0

_SOURCE_DECAY = """\
begin parameters
    1 p   1.0{k}
    2 c1  2e18{k}
    3 c2  1.0{k}
    4 a   p*c1{e}
    5 b   p*c2{e}
end parameters
begin species
    1 $S() 1.0
    2 X()  0
end species
begin reactions
    1 1 2 a  #_R1
    2 2 0 b  #_R2
end reactions
begin groups
    1 S_tot  1
    2 X_tot  2
end groups
"""

BARE = _SOURCE_DECAY.format(k="", e="")
ANNOTATED = _SOURCE_DECAY.format(k="    # Constant", e="   # ConstantExpression")
# A .net whose annotation is simply wrong. Nothing rejects it, so the expression
# has to win outright — honouring the comment when it is present would leave this
# file with the pre-#181 zero.
MISLABELLED = _SOURCE_DECAY.format(k="    # Constant", e="   # Constant")


def _write(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _dX_dp(times: np.ndarray) -> np.ndarray:
    """Closed-form dX/dp for the model above."""
    return C1 * times * np.exp(-P * C2 * times)


def _sensitivity_of(net: str, param: str, species: int, times) -> np.ndarray:
    sim = bngsim.Simulator(bngsim.Model.load(net), method="ode", sensitivity_params=[param])
    result = sim.run(
        t_span=(float(times[0]), float(times[-1])),
        n_points=len(times),
        sample_times=list(times),
        timeout=120.0,
    )
    return np.asarray(result.sensitivities)[:, species, 0]


# ─── The kind annotation is not the source of truth ──────────────────────


@pytest.mark.parametrize(
    "text", [BARE, ANNOTATED, MISLABELLED], ids=["bare", "annotated", "lying"]
)
def test_parameter_kinds_are_read_off_the_expressions(tmp_path, text):
    """``a``/``b`` reference ``p``; ``p``/``c1``/``c2`` reference nothing."""
    parsed = _parse_net_file(_write(tmp_path, "m.net", text))
    kinds = {name: is_const for _, name, _, is_const in parsed["parameters"]}
    assert kinds == {"p": True, "c1": True, "c2": True, "a": False, "b": False}


@pytest.mark.parametrize("text", [BARE, MISLABELLED], ids=["bare", "lying"])
def test_stripping_the_annotations_does_not_change_the_emitted_c(tmp_path, text):
    """The strongest form: the comments carry no information the emitter needs."""
    annotated = generate_sens_rhs_c(_write(tmp_path, "annotated.net", ANNOTATED))
    assert annotated is not None
    assert generate_sens_rhs_c(_write(tmp_path, "other.net", text)) == annotated


def test_unannotated_net_reports_the_true_sensitivity(tmp_path):
    """The issue's measurement: ``[0, 0, 0]`` where the answer is not zero."""
    times = np.array([0.0, 1.0, 5.0])
    exact = _dX_dp(times)

    bare = _sensitivity_of(_write(tmp_path, "bare.net", BARE), "p", 1, times)
    annotated = _sensitivity_of(_write(tmp_path, "annotated.net", ANNOTATED), "p", 1, times)

    assert np.any(bare != 0.0), "dX/dp came back identically zero (GH #181)"
    np.testing.assert_allclose(bare, exact, rtol=1e-5)
    np.testing.assert_allclose(bare, annotated, rtol=1e-12, atol=0.0)


# ─── …and a literal-valued expression is still a leaf ────────────────────

# BNG2.pl writes `# Constant` here, because `ln(2)/10` names no other parameter.
# 628 parameter lines in this tree look like this — `2*asin(1)`, `37+273.15`,
# `6.022e23*1e-15` — and a float-parse rule would call every one of them derived.
_LITERAL_ARITHMETIC = """\
begin parameters
    1 kdeg   ln(2)/10  # Constant
end parameters
begin species
    1 A() 100
end species
begin reactions
    1 1 0 kdeg  #_R1
end reactions
begin groups
    1 A_tot  1
end groups
"""


def test_literal_arithmetic_parameter_stays_a_differentiation_leaf(tmp_path):
    """``kdeg = ln(2)/10`` has no primary behind it, so it *is* the primary.

    The bound on the fix, and the reason it reads the parameter *references*
    rather than the value syntax: a rule that called this one derived would agree
    with BNG2.pl nowhere, and would push 54 models in this tree through a
    different sensitivity RHS to answer a question none of them asked.
    """
    net = _write(tmp_path, "literal.net", _LITERAL_ARITHMETIC)

    parsed = _parse_net_file(net)
    assert [(n, is_const) for _, n, _, is_const in parsed["parameters"]] == [("kdeg", True)]

    times = np.array([0.0, 5.0, 20.0])
    measured = _sensitivity_of(net, "kdeg", 0, times)
    kdeg = np.log(2.0) / 10.0
    np.testing.assert_allclose(measured, -100.0 * times * np.exp(-kdeg * times), rtol=1e-5)


# ─── Every shipped .net emits the same C without its kind comments ───────

_KIND_COMMENT_RE = re.compile(r"\s*#\s*Constant(Expression)?\s*$")


def _strip_kind_comments(text: str) -> str:
    """Remove trailing kind annotations from parameter lines only.

    A reaction's ``#_R1`` and a species' comment must survive: those the parser is
    entitled to read.
    """
    out, in_parameters = [], False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("begin "):
            in_parameters = stripped.split()[1] == "parameters"
        elif stripped.startswith("end "):
            in_parameters = False
        elif in_parameters:
            eol = "\n" if line.endswith("\n") else ""
            line = _KIND_COMMENT_RE.sub("", line.rstrip("\n")) + eol
        out.append(line)
    return "".join(out)


_ANNOTATED_FIXTURES = sorted(
    p for p in DATA_DIR.glob("*.net") if "# Constant" in p.read_text(encoding="utf-8")
)


@pytest.mark.skipif(not _ANNOTATED_FIXTURES, reason="tests/data .net fixtures not present")
@pytest.mark.parametrize("net", _ANNOTATED_FIXTURES, ids=lambda p: p.stem)
def test_shipped_fixture_emits_the_same_c_without_its_kind_comments(tmp_path, net):
    """The no-op half of the change, over every annotated .net this repo ships.

    Whatever ``generate_sens_rhs_c`` does with a BNG2.pl file — emit C, decline
    with ``None``, or raise — it must do the identical thing to the same file with
    the kind annotations deleted.
    """
    stripped = _strip_kind_comments(net.read_text(encoding="utf-8"))
    assert stripped != net.read_text(encoding="utf-8"), "fixture lost its annotations"
    other = _write(tmp_path, net.name, stripped)

    def outcome(path: str):
        try:
            return ("ok", generate_sens_rhs_c(path))
        except Exception as exc:  # noqa: BLE001 — an equal failure is still equal
            return ("raised", f"{type(exc).__name__}: {exc}")

    assert outcome(other) == outcome(str(net))
