"""Every BNGsim error carries a code, details and remedies (issue #1702).

A program that drives BNGsim (a tool server, an agent's retry loop, a benchmark
that sorts outcomes) branches on ``e.code`` instead of matching the message,
which changes between releases. These tests hold the registry and the page
generated from it together, and check that each of the failures the issue
names carries the code a caller would branch on, with the details and remedies
it promises, while still being the built-in type an existing ``except`` names.
"""

from __future__ import annotations

import ast
import inspect
import json
import pickle
import sys
from pathlib import Path

import bngsim
import pytest
from bngsim import _error_codes, _exceptions
from bngsim._error_codes import CODES, GENERIC, REMEDY_KINDS

_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE = Path(bngsim.__file__).resolve().parent

_ERROR_CLASSES = [
    cls
    for _, cls in inspect.getmembers(_exceptions, inspect.isclass)
    if issubclass(cls, bngsim.BngsimError)
]


def _check_shape(e: bngsim.BngsimError) -> None:
    """What every error promises: a registered code, plain-data fields, remedies of
    registered kinds with their keys, and a to_dict() that JSON can write."""
    assert e.code in CODES, e.code
    assert isinstance(e.details, dict)
    for remedy in e.remedies:
        _, keys = REMEDY_KINDS[remedy["kind"]]
        assert all(k in remedy for k in keys), remedy
    d = json.loads(json.dumps(e.to_dict()))
    assert d["code"] == e.code and d["message"] == str(e) and d["type"] == type(e).__name__
    assert d["docs_url"].endswith("#" + _error_codes.anchor(e.code))


# ─── The registry ─────────────────────────────────────────────────────────────


def test_every_class_default_is_registered_and_names_its_class():
    for cls in _ERROR_CLASSES:
        assert cls.default_code in CODES, cls.__name__
    # Each code's "raised as" class exists, so the page names real types.
    for c in CODES.values():
        assert hasattr(bngsim, c.error) or hasattr(_exceptions, c.error), c.error
    assert set(CODES) >= GENERIC


def test_every_code_the_source_passes_is_registered():
    """A ``code="..."`` keyword anywhere in the package, and each solver flag's
    code, is in the registry, so no error carries a code the page lacks."""
    used = set(_error_codes.SOLVER_FLAGS.values()) | {"solver.failed", "env.no_compiler"}
    for path in _PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "code"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                used.add(node.value.value)
    # The model's own not-found errors build theirs from the kind of name.
    used |= {"param.not_found", "species.not_found"}
    missing = sorted(c for c in used if c not in CODES and "." in c)
    assert not missing, missing
    assert len(used) > 15


def test_anchors_are_unique():
    anchors = [_error_codes.anchor(c) for c in CODES]
    assert len(anchors) == len(set(anchors))


def test_the_reference_page_is_current():
    page = _ROOT / _error_codes.PAGE
    if not page.exists():
        pytest.skip("docs/ is not in an installed wheel")
    assert page.read_text(encoding="utf-8") == _error_codes.reference_page(), (
        "docs/reference/error-codes.md differs from the registry: run "
        "`python scripts/error_codes_page.py`"
    )


# ─── The failures the issue names ─────────────────────────────────────────────

_ALGEBRAIC = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
<model id="m"><listOfCompartments>
<compartment id="c" size="1" constant="true"/></listOfCompartments>
<listOfSpecies>
<species id="A" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
<species id="B" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
</listOfSpecies><listOfRules><algebraicRule>
<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><minus/><ci>A</ci><ci>B</ci></apply>
</math></algebraicRule></listOfRules></model></sbml>"""


@pytest.fixture
def binding():
    return bngsim.examples.load("binding")  # parameters kf, kr, A0, B0; observable C


def test_an_unsupported_construct_names_it():
    with pytest.raises(bngsim.ModelError) as info:
        bngsim.Model.from_sbml_string(_ALGEBRAIC)
    e = info.value
    # The loader wraps the refusal ("Failed to load SBML string: ..."), and the
    # wrapper carries the refusal's code and details.
    assert e.code == "model.unsupported_construct"
    assert e.details == {"constructs": [{"construct": "AlgebraicRule", "where": "0 = A - B"}]}
    assert e.remedies == []  # no fix BNGsim can name: the model cannot run as written
    _check_shape(e)


def test_an_unknown_parameter_gives_its_close_matches(binding):
    with pytest.raises(bngsim.ParameterError) as info:
        binding.set_param("kff", 2.0)
    e = info.value
    assert e.code == "param.not_found"
    assert e.details == {"name": "kff", "close_matches": ["kf"], "is_a": None}
    assert e.remedies == [{"kind": "use_name", "name": "kf"}]
    _check_shape(e)


@pytest.mark.parametrize(
    ("call", "code", "is_a", "remedy"),
    [
        (lambda m: m.set_param("A(b)", 1.0), "param.not_found", "species",
         {"kind": "use_call", "call": "set_concentration", "name": "A(b)"}),
        (lambda m: m.set_concentration("kf", 1.0), "species.not_found", "parameter",
         {"kind": "use_call", "call": "set_param", "name": "kf"}),
        (lambda m: m.set_param("C", 1.0), "param.not_found", "observable", None),
    ],
)  # fmt: skip
def test_a_name_of_another_kind_says_what_it_is(binding, call, code, is_a, remedy):
    with pytest.raises(bngsim.BngsimError) as info:
        call(binding)
    e = info.value
    assert (e.code, e.details["is_a"]) == (code, is_a)
    assert (remedy in e.remedies) if remedy else e.remedies == []
    _check_shape(e)


def test_a_failure_after_the_lookup_is_not_a_missing_name(binding):
    """A parameter the model has is not ``param.not_found``, whatever failed."""
    assert binding._not_found("kf", "parameter") == {}


def test_too_much_accuracy_is_the_solver_flag(binding):
    sim = bngsim.Simulator(binding, jacobian="fd")
    with pytest.raises(bngsim.SimulationError) as info:
        sim.run(t_span=(0, 10), n_points=3, rtol=1e-20, start="initial")
    e = info.value
    assert e.code == "solver.too_much_accuracy"
    assert e.details == {"flag": "CV_TOO_MUCH_ACC"}
    assert e.remedies == [{"kind": "loosen_tolerance"}]
    _check_shape(e)


@pytest.mark.parametrize(
    ("flag", "code"),
    [
        ("CV_TOO_MUCH_WORK", "solver.too_much_work"),
        ("CV_ERR_FAILURE", "solver.error_test_failure"),
        ("CV_CONV_FAILURE", "solver.convergence_failure"),
        ("CV_RHSFUNC_FAIL", "solver.failed"),
    ],
)
def test_each_solver_flag_gets_its_code(flag, code):
    message = f"Simulation failed: CVODE integration failed at t=1.000000 with flag=-3 ({flag})."
    e = bngsim.SimulationError(message)
    assert (e.code, e.details) == (code, {"flag": flag})
    _check_shape(e)
    # A subclass with a code of its own keeps it.
    assert bngsim.SparseJacobianUnavailableError(message).code == (
        "solver.sparse_jacobian_unavailable"
    )
    # A message with no flag is a generic failure.
    assert bngsim.SimulationError("Simulation failed: NaN").code == "run.failed"


def test_an_unknown_suffix_lists_the_known_ones():
    with pytest.raises(bngsim.ModelError) as info:
        bngsim.Model.load("notes.txt")
    e = info.value
    assert e.code == "model.unknown_format"
    assert e.details["suffix"] == ".txt"
    assert ".net" in e.details["expected"] and ".xml" in e.details["expected"]
    _check_shape(e)


@pytest.mark.skipif(not bngsim.HAS_NFSIM, reason="this build has no NFsim")
def test_a_missing_xml_path_is_a_usage_error(binding):
    with pytest.raises(ValueError) as info:  # the type it had before it was typed
        bngsim.Simulator(binding, method="nf")
    e = info.value
    assert isinstance(e, bngsim.UsageError)
    assert e.code == "usage.missing_argument"
    assert e.details == {"argument": "xml_path", "method": "nf"}
    assert e.remedies == [{"kind": "pass_argument", "argument": "xml_path"}]
    _check_shape(e)


def test_a_missing_package_gives_the_install_command(binding, monkeypatch):
    result = bngsim.Simulator(binding).run(t_span=(0, 1), n_points=2)
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError) as info:  # the type it had before it was typed
        _ = result.dataframe
    e = info.value
    assert isinstance(e, bngsim.DependencyError)
    assert e.code == "env.missing_dependency"
    assert e.remedies == [
        {
            "kind": "install_package",
            "package": "pandas",
            "command": "pip install 'bngsim[pandas]'",
            "extra": "pandas",
        }
    ]
    _check_shape(e)


def test_a_missing_file_gives_its_path(tmp_path):
    missing = tmp_path / "missing.xml"
    with pytest.raises(FileNotFoundError) as info:  # the type it had before
        bngsim.Model.load(missing)
    e = info.value
    assert isinstance(e, bngsim.MissingFileError)
    assert e.code == "file.not_found"
    assert e.details == {"path": str(missing)}
    assert str(e) == f"SBML file not found: {missing}"  # the message is unchanged
    _check_shape(e)


def test_a_missing_compiler_is_told_from_a_broken_one():
    missing = bngsim.CodegenToolchainError(
        "Failed to build the analytical sensitivity RHS required for forward sensitivity "
        "(CodegenToolchainError: C compiler not found: 'cc' is not an executable on PATH)."
    )
    assert missing.code == "env.no_compiler"
    assert missing.remedies == [{"kind": "install_compiler"}]
    broken = bngsim.CodegenToolchainError("ld: cannot read libSystem.tbd")
    assert broken.code == "env.broken_toolchain" and broken.remedies == []
    _check_shape(missing)
    _check_shape(broken)


def test_the_typed_refusals_carry_their_class_codes():
    assert bngsim.SimulationTimeout("slow", timeout=1.0, elapsed=2.5).to_dict()["details"] == {
        "timeout": 1.0,
        "elapsed": 2.5,
    }
    assert bngsim.StopConditionMet("stop", condition="A < 1").details == {"condition": "A < 1"}
    for cls in _ERROR_CLASSES:
        if cls in (
            bngsim.SsaValidationError,
            bngsim.SimulationTimeout,
            bngsim.StopConditionMet,
            bngsim.MissingFileError,
        ):
            continue
        assert cls("x").code == cls.default_code, cls.__name__


# ─── How a code travels ───────────────────────────────────────────────────────


def test_a_wrapper_carries_a_specific_code_and_not_a_generic_one():
    def wrapped(inner: Exception) -> bngsim.ModelError:
        try:
            try:
                raise inner
            except Exception as e:
                raise bngsim.ModelError(f"Failed to load: {e}") from e
        except bngsim.ModelError as outer:
            return outer

    specific = wrapped(bngsim.UnderSpecifiedModelError("x"))
    assert specific.code == "model.unset_symbol"
    assert wrapped(bngsim.SimulationError("y")).code == "model.error"  # generic: not carried
    assert wrapped(ValueError("z")).code == "model.error"
    own = bngsim.ModelError("w", code="model.unknown_format")
    own.__cause__ = bngsim.UnderSpecifiedModelError("x")
    assert own.code == "model.unknown_format" and own.details == {}


def test_the_fields_survive_pickling():
    """A worker process's error reaches its parent with its code and details."""
    for e in (
        bngsim.ParameterError("p", code="param.not_found", details={"name": "k"}),
        bngsim.MissingFileError("SBML file not found: x.xml", "x.xml"),
        bngsim.SimulationError("CVODE failed with flag=-2 (CV_TOO_MUCH_ACC)."),
    ):
        back = pickle.loads(pickle.dumps(e))
        assert back.to_dict() == e.to_dict()
