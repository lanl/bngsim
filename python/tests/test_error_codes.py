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


def _subclasses(cls: type) -> list[type]:
    found = [cls]
    for sub in cls.__subclasses__():
        found += [c for c in _subclasses(sub) if c not in found]
    return found


import bngsim.coupling  # noqa: E402, F401  (defines ConservationError)

#: Every BngsimError class in the package, wherever it is defined.
_ERROR_CLASSES = [c for c in _subclasses(bngsim.BngsimError) if c.__module__.startswith("bngsim")]


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
    assert bngsim.coupling.ConservationError in _ERROR_CLASSES
    assert bngsim.coupling.ConservationError("x").code == "coupling.conservation"
    for cls in _ERROR_CLASSES:
        assert cls.default_code in CODES, cls.__name__
    # Each code's "raised as" class exists, so the page names real types.
    for c in CODES.values():
        assert hasattr(bngsim, c.error) or hasattr(_exceptions, c.error), c.error
    assert set(CODES) >= GENERIC


def _string_constants() -> set[str]:
    """Every string constant in the package outside the registry module."""
    found: set[str] = set()
    for path in _PACKAGE.rglob("*.py"):
        if path.name == "_error_codes.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.add(node.value)
    return found


def test_every_registered_code_can_be_raised():
    """The other direction: each code in the registry is some class's default, a
    solver flag's, or a string the package spells, so the page lists no code
    nothing raises."""
    reachable = {c.default_code for c in _ERROR_CLASSES}
    reachable |= set(_error_codes.SOLVER_FLAGS.values())
    reachable |= _string_constants() & set(CODES)
    assert set(CODES) - reachable == set()


def _notice_keywords(tree: ast.AST) -> set[int]:
    """The ``code=`` keywords of ``_notices.*`` calls: a warning's code, which
    ``bngsim._notices.CODES`` registers and ``test_result_notices`` checks."""
    return {
        id(k)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_notices"
        for k in node.keywords
    }


def test_every_code_the_source_passes_is_registered():
    """A ``code="..."`` keyword anywhere in the package, and each solver flag's
    code, is in the registry, so no error carries a code the page lacks."""
    used = set(_error_codes.SOLVER_FLAGS.values())
    for path in _PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        notices = _notice_keywords(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "code"
                and id(node) not in notices
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                used.add(node.value.value)
    missing = sorted(c for c in used if c not in CODES and "." in c)
    assert not missing, missing
    assert len(used) > 15


def test_an_error_code_is_never_a_warnings_code():
    """A program that branches on ``code`` reads errors' and a Result's notices'
    codes alike (issue #1703), so no code names both."""
    from bngsim import _notices

    assert set(CODES).isdisjoint(_notices.CODES)


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
    """The flags as cvode_failure_message in src/cvode_simulator.cpp writes them;
    the real path is test_too_much_accuracy_is_the_solver_flag."""
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


@pytest.mark.skipif(not bngsim.HAS_NFSIM, reason="NFsim not compiled in")
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


def test_a_loose_species_match_agrees_with_the_message():
    """The details say what the message says: for ``A`` on a model whose species
    is ``A()``, the species to set, by the model's own spelling."""
    model = bngsim.examples.load("decay")  # species A(), observable A, parameter k
    with pytest.raises(bngsim.ParameterError) as info:
        model.set_param("A", 2.0)
    e = info.value
    assert "The species 'A()' is close" in str(e)
    assert e.details == {"name": "A", "close_matches": [], "is_a": "species"}
    assert e.remedies == [{"kind": "use_call", "call": "set_concentration", "name": "A()"}]
    with pytest.raises(bngsim.ModelError) as info:
        model.set_concentration("a", 2.0)
    assert info.value.details["close_matches"] == ["A()"]
    assert info.value.remedies == [{"kind": "use_name", "name": "A()"}]


def test_an_ssa_refusal_survives_pickling():
    from bngsim._ssa_validation import SsaIssue

    issue = SsaIssue(severity="error", code="fast_reaction", message="m", location="r1")
    e = bngsim.SsaValidationError([issue], override_attempted=True)
    back = pickle.loads(pickle.dumps(e))
    assert back.to_dict() == e.to_dict() and back.override_attempted
    assert back.details == {
        "issues": [{"code": "fast_reaction", "severity": "error", "location": "r1"}]
    }


def test_editing_what_an_error_gives_does_not_change_it():
    e = bngsim.ParameterError("p", code="param.not_found", details={"close_matches": ["k"]})
    e.to_dict()["details"]["close_matches"].clear()
    e.details["close_matches"].append("x")
    assert e.details == {"close_matches": ["k"]}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-specific: empty PATH")
def test_a_run_with_no_compiler_on_path_is_env_no_compiler(tmp_path):
    """The real path, not a message built by hand: forward sensitivities on a
    machine with no compiler and an empty codegen cache."""
    import os
    import subprocess

    (tmp_path / "bin").mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("CC", "BNGSIM_CODEGEN_JIT")}
    env["PATH"] = str(tmp_path / "bin")
    env["BNGSIM_CODEGEN_CACHE_DIR"] = str(tmp_path / "cache")
    script = (
        "import json, bngsim\n"
        "m = bngsim.examples.load('decay')\n"
        "try:\n"
        "    bngsim.Simulator(m, sensitivity_params=['k']).run(t_span=(0, 1), n_points=2)\n"
        "except bngsim.BngsimError as e:\n"
        "    print(json.dumps(e.to_dict()))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    ).stdout
    d = json.loads(out.strip().splitlines()[-1])
    assert d["type"] == "CodegenToolchainError"
    assert d["code"] == "env.no_compiler"
    assert d["remedies"] == [{"kind": "install_compiler"}]


def test_a_stall_is_its_own_code_with_a_tolerance_remedy():
    """A collapsed step reaches the caller as "CVODE made no progress", not as a
    flag: the engine retries CV_TOO_MUCH_WORK while the solver advances, so a
    larger max_steps is no remedy for it."""
    e = bngsim.SimulationError(
        "Simulation failed: CVODE made no progress during the run: an entire max_steps "
        "batch advanced the internal time not at all, at t=1 with step size h=0."
    )
    assert e.code == "solver.stalled"
    assert e.remedies == [{"kind": "loosen_tolerance"}]
    assert bngsim.SimulationError("x flag=-1 (CV_TOO_MUCH_WORK).").remedies == []


def test_a_class_with_its_own_code_keeps_it_when_raised_from_another():
    def raised_from(outer: bngsim.BngsimError, inner: bngsim.BngsimError):
        outer.__cause__ = inner
        return outer

    toolchain = bngsim.CodegenToolchainError("C compiler not found: 'cc'")
    declined = raised_from(bngsim.CodegenDeclinedError("declined"), toolchain)
    assert (declined.code, declined.remedies) == ("codegen.declined", [])
    timeout = raised_from(
        bngsim.SimulationTimeout("slow", timeout=1.0, elapsed=2.0),
        bngsim.ParameterError("p", code="param.not_found", details={"name": "k"}),
    )
    assert timeout.code == "run.timeout"
    assert timeout.details == {"timeout": 1.0, "elapsed": 2.0}


def test_set_params_names_each_unknown_parameter(binding):
    with pytest.raises(bngsim.ParameterError) as info:
        binding.set_params({"kff": 1.0, "kf": 2.0})
    e = info.value
    assert e.code == "param.not_found"
    assert e.details == {"names": [{"name": "kff", "close_matches": ["kf"], "is_a": None}]}
    assert e.remedies == [{"kind": "use_name", "name": "kf"}]
    assert binding.get_param("kf") != 2.0  # nothing was set


def test_a_dependency_error_reads_as_the_import_error_it_was():
    message = "pandas is required for Result.dataframe. Install with: pip install bngsim[pandas]"
    e = bngsim.DependencyError(message)
    old = ImportError(message)
    assert (str(e), e.args, e.msg, e.name) == (str(old), old.args, old.msg, old.name)


def test_a_missing_antimony_gives_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "antimony", None)
    with pytest.raises(ImportError) as info:
        bngsim.Model.from_antimony_string("model m\n A -> ; k*A\n A = 1\n k = 1\nend")
    e = info.value
    assert isinstance(e, bngsim.DependencyError)
    assert e.code == "env.missing_dependency"
    assert e.remedies[0]["command"] == "pip install 'bngsim[antimony]'"
    _check_shape(e)


# ─── A failed run's notices (issue #1857) ─────────────────────────────────────

_BLOWUP = """begin parameters
    1 k 1
end parameters
begin species
    1 X() 1
end species
begin reactions
    1 1 1,1 k
end reactions
begin groups
    1 Xtot 1
end groups
"""


def test_a_failed_run_keeps_its_notices_on_the_error(tmp_path):
    """X grows as e^t, so a long window with few steps fails. The second run
    starts from the first's end state and retries on a difference-quotient
    Jacobian before it fails; with every warning filtered out, the error alone
    says both."""
    import warnings

    net = tmp_path / "blowup.net"
    net.write_text(_BLOWUP)
    sim = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode")
    sim.run(t_span=(0, 0.5), n_points=3)
    with warnings.catch_warnings(), pytest.raises(bngsim.SimulationError) as info:
        warnings.simplefilter("ignore")
        sim.run(t_span=(0, 1e6), n_points=3, max_steps=50)
    e = info.value
    codes = [n["code"] for n in e.notices]
    assert codes == ["start.continues_last_run", "jacobian.difference_quotient_retry"]
    assert e.to_dict()["notices"] == e.notices
    assert all(set(n) == {"category", "code", "message"} for n in e.notices)
    assert pickle.loads(pickle.dumps(e)).notices == e.notices
    json.dumps(e.to_dict())


def _noting(code: str, raises: BaseException | None = None):
    from bngsim import _notices

    @_notices.collected
    def call(inner=None):
        _notices.note("notice", code, code)
        if inner is not None:
            inner()
        if raises is not None:
            raise raises
        return None

    return call


def test_nested_calls_give_each_notice_once_in_order():
    """The innermost collector takes a notice, so an error raised through two
    collected calls has the inner call's notices, then the outer's, each once."""
    inner = _noting("sens.chunk_split", bngsim.SimulationError("failed"))
    outer = _noting("start.rows_from_initial")
    with pytest.raises(bngsim.SimulationError) as info:
        outer(inner)
    assert [n["code"] for n in info.value.notices] == [
        "sens.chunk_split",
        "start.rows_from_initial",
    ]


def test_an_error_that_is_not_bngsims_is_left_alone():
    call = _noting("sens.chunk_split", KeyError("bug"))
    with pytest.raises(KeyError) as info:
        call()
    assert not hasattr(info.value, "notices")


def test_an_error_raised_from_a_failed_run_carries_its_notices():
    """A wrapper raised from a run's error has that run's notices, first."""
    call = _noting("sens.chunk_split", bngsim.SimulationError("failed"))
    try:
        try:
            call()
        except bngsim.SimulationError as cause:
            raise bngsim.ModelError("Failed to run the protocol") from cause
    except bngsim.ModelError as e:
        assert [n["code"] for n in e.notices] == ["sens.chunk_split"]
        assert e.to_dict()["notices"] == e.notices


def test_an_error_outside_a_run_has_no_notices():
    e = bngsim.ParameterError("p")
    assert e.notices == []
    assert e.to_dict()["notices"] == []


def test_an_ssa_validation_error_keeps_its_notices_through_pickling():
    from bngsim._ssa_validation import SsaIssue

    issue = SsaIssue(code="x", severity="error", location="r1", message="m")
    e = bngsim.SsaValidationError([issue])
    e._add_notices([{"category": "notice", "code": "ssa.rounded_counts", "message": "m"}])
    back = pickle.loads(pickle.dumps(e))
    assert back.to_dict() == e.to_dict()
    assert back.issues == e.issues


# ─── Notice codes on the reference page (issue #1858) ─────────────────────────


def test_every_notice_code_says_what_to_do():
    from bngsim import _notices

    assert list(_notices.ACTIONS) == list(_notices.CODES)
    assert all(a.strip() for a in _notices.ACTIONS.values())


def test_the_page_lists_every_notice_code():
    from bngsim import _notices

    page = _error_codes.reference_page()
    for code, meaning in _notices.CODES.items():
        assert f"| `{code}` | {meaning} | {_notices.ACTIONS[code]} |" in page
