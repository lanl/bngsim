"""``Result.notices`` (issue #1703): every warning and fallback of a run, on its Result.

A run's complaints left by routes none of which is the Result: a Python warning,
which a filter drops and the default filter shows once per line; a line logged
through the ``bngsim`` logger, which a level drops; and SUNDIALS' own lines on
standard error. A Result read later, or by a program that never saw the console,
did not say that its Jacobian had been retried on a difference quotient or that
its tolerance was above a species it reports.

Every site in the package that warns or logs a warning either goes through
``bngsim._notices``, which records the notice with the call being made before it
warns or logs as before, or is named in ``NOT_OF_A_RUN`` below with why no run
reaches it. A new site has to choose.
"""

from __future__ import annotations

import ast
import logging
import warnings
from pathlib import Path

import bngsim
import pytest
from bngsim import Model, Result, Simulator, _notices
from bngsim import _simulator as simulator_module

PACKAGE = Path(bngsim.__file__).resolve().parent

#: (file under python/bngsim, function) -> why no run reaches its warnings. A
#: function of ``*`` names every site in the file.
NOT_OF_A_RUN: dict[tuple[str, str], str] = {
    ("_build_provenance.py", "warn_if_stale"): "the import's check of the build",
    ("_build_provenance.py", "enforce"): "the import's check of the build",
    ("_codegen.py", "prepare_codegen"): "a deprecated call, not a run",
    ("_model.py", "Model.add_table_function"): "an edit of the model",
    ("_model.py", "_warn_dropped_state_changes"): "a BNGL load",
    ("_net_reader.py", "parse_net_file"): "a .net load",
    ("_protocol_run.py", "*"): (
        "a protocol's reading of its actions (an argument it ignores, a poplevel it "
        "supplies, an empty interval), not of a run; each run it makes records its own"
    ),
    ("_result.py", "Result._check_missing_volume_record"): "as_roadrunner(), reading a result",
    ("_result.py", "Result.save"): "writing a file",
    ("_result.py", "_identifiability_from_fim"): "reading a result's sensitivities",
    ("_sbml_loader.py", "*"): "an SBML or Antimony load",
    ("_simulator.py", "Simulator.__init__"): (
        "building the Simulator: what it says then is not of any one run, and its "
        "SSA validation reads the state at construction, not a run's start. A "
        "sensitivity run_batch builds one for rows of another attachment inside the "
        "call; what that says of the call's arguments was said when this one was built"
    ),
    ("_simulator.py", "Simulator._note_ss_jacobian_retry"): (
        "steady_state(), whose SteadyStateResult has no record yet"
    ),
    ("_simulator.py", "Simulator._warn_about_pure_sinks"): "steady_state(), as above",
    ("_simulator.py", "Simulator._warn_about_ss_sensitivity"): "steady_state(), as above",
    ("_simulator.py", "Simulator._set_clock"): "set_time(), which runs nothing",
    ("convert/*", "*"): "a format conversion, or a SED-ML or BNGL protocol read",
}

_WARNERS = {"warnings", "_warnings"}
_LOGGERS = {"logger", "_logger", "log", "_log"}


class _Calls(ast.NodeVisitor):
    """Each call of one file, with the function it is in."""

    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.stack: list[str] = []
        self.out: list[tuple[str, str, int, ast.Call]] = []

    def visit_FunctionDef(self, node: ast.AST) -> None:
        self.stack.append(node.name)  # type: ignore[attr-defined]
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        self.out.append((self.rel, ".".join(self.stack), node.lineno, node))
        self.generic_visit(node)


def _calls() -> list[tuple[str, str, int, ast.Call]]:
    """Every call in the package, as (file, function, line, call)."""
    out: list[tuple[str, str, int, ast.Call]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        walk = _Calls(path.relative_to(PACKAGE).as_posix())
        walk.visit(ast.parse(path.read_text(encoding="utf-8")))
        out.extend(walk.out)
    return out


def _is_warning_site(call: ast.Call) -> bool:
    f = call.func
    if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)):
        return False
    return (f.value.id in _WARNERS and f.attr in ("warn", "warn_explicit")) or (
        f.value.id in _LOGGERS and f.attr == "warning"
    )


def _exempt(rel: str, func: str) -> bool:
    folder = rel.split("/")[0] + "/*" if "/" in rel else None
    return any(key in NOT_OF_A_RUN for key in ((rel, func), (rel, "*"), (folder or "", "*")))


def test_every_warning_in_the_package_is_a_notice_or_says_why_not() -> None:
    unrouted = [
        f"{rel}:{line} in {func or '<module>'}"
        for rel, func, line, call in _calls()
        if rel != "_notices.py" and _is_warning_site(call) and not _exempt(rel, func)
    ]
    assert not unrouted, (
        "these warn or log past bngsim._notices, so a Result never hears of them: "
        "route each through _notices.warn or _notices.log with a code, or name it "
        f"in NOT_OF_A_RUN with why no run reaches it: {unrouted}"
    )


def test_every_exemption_names_a_site_that_is_there() -> None:
    sites = {(rel, func) for rel, func, _, call in _calls() if _is_warning_site(call)}
    files = {rel for rel, _ in sites}
    folders = {rel.split("/")[0] + "/*" for rel in files if "/" in rel}
    stale = [
        key
        for key in NOT_OF_A_RUN
        if not (key in sites or (key[1] == "*" and (key[0] in files or key[0] in folders)))
    ]
    assert not stale, f"NOT_OF_A_RUN names sites that no longer warn: {stale}"


def _codes_used() -> set[str]:
    used: set[str] = set()
    for _, _, _, call in _calls():
        f = call.func
        if not (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id == "_notices"
            and f.attr in ("warn", "log", "note")
        ):
            continue
        if f.attr == "warn":
            values = [k.value for k in call.keywords if k.arg == "code"]
        else:
            values = [call.args[1]] if len(call.args) > 1 else []
        assert len(values) == 1 and isinstance(values[0], ast.Constant), (
            f"a notice whose code is not written at its site: {ast.unparse(call)[:80]}"
        )
        used.add(values[0].value)
    return used


def test_every_code_used_is_listed_and_every_code_listed_is_used() -> None:
    used = _codes_used()
    assert used - set(_notices.CODES) == set(), "codes used and not in _notices.CODES"
    assert set(_notices.CODES) - used == set(), "codes in _notices.CODES no site uses"
    assert list(_notices.CODES) == sorted(_notices.CODES)


# ─── The notices a run records ─────────────────────────────────────────

# X starts at 0 and grows at 1e-12 a unit of time: it stays below the default
# atol, 1e-8, which nothing scales it under (issues #1361, #1420).
SOURCE = """\
begin parameters
    1 k 1e-12
end parameters
begin species
    1 X() 0
end species
begin reactions
    1 0 1 k
end reactions
begin groups
    1 X 1
end groups
"""

AB = """\
begin parameters
    1 kf 0.5
    2 A0 10
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 kf
end reactions
begin groups
    1 Atot 1
end groups
"""


def _net(tmp_path: Path, name: str, text: str) -> Model:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return Model.from_net(path)


def _codes(result: Result) -> list[str]:
    return [n["code"] for n in result.notices]


def test_a_warning_the_filter_shows_once_is_on_every_run_it_concerns(tmp_path: Path) -> None:
    sim = Simulator(_net(tmp_path, "source.net", SOURCE))
    with warnings.catch_warnings(record=True) as shown:
        # The default action shows a warning once per line that gives it.
        warnings.simplefilter("default")
        runs = [sim.run(t_span=(0, 1), n_points=3, start="initial") for _ in range(2)]
    assert [w.category for w in shown] == [bngsim.AtolScaleWarning]
    for r in runs:
        (notice,) = r.notices
        assert notice["category"] == "AtolScaleWarning"
        assert notice["code"] == "atol.below_scale"
        assert "'X()'" in notice["message"]


def test_a_warning_filtered_out_is_still_on_the_result(tmp_path: Path) -> None:
    sim = Simulator(_net(tmp_path, "source.net", SOURCE))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sim.run(t_span=(0, 1), n_points=3)
    assert _codes(r) == ["atol.below_scale"]


def test_a_warning_made_an_error_still_raises(tmp_path: Path) -> None:
    sim = Simulator(_net(tmp_path, "source.net", SOURCE))
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.AtolScaleWarning)
        with pytest.raises(bngsim.AtolScaleWarning):
            sim.run(t_span=(0, 1), n_points=3)


def test_a_run_that_warns_of_nothing_has_no_notices(tmp_path: Path) -> None:
    r = Simulator(_net(tmp_path, "ab.net", AB)).run(t_span=(0, 1), n_points=3)
    assert r.notices == []


# S empties at t = 2, where d/dS sqrt(S) is infinite (GH #176, #867).
SINK = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="sink">
    <listOfCompartments><compartment id="V" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="V" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="1" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="R1" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><apply><root/><ci>S</ci></apply></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def test_a_logged_fallback_is_on_the_result_whatever_the_logging_level() -> None:
    """The issue's example: the GH #176 retry is a logged notice, and a run
    that went on to succeed kept no trace of it."""
    sim = Simulator(Model.from_sbml_string(SINK))
    assert sim.jacobian_strategy == "analytical"
    log = logging.getLogger("bngsim")
    level = log.level
    log.setLevel(logging.CRITICAL)
    try:
        r = sim.run(t_span=(0, 40), n_points=5)
    finally:
        log.setLevel(level)
    (notice,) = r.notices
    assert notice["category"] == "log"
    assert notice["code"] == "jacobian.difference_quotient_retry"
    assert notice["message"].startswith("analytical Jacobian: CVODE integration failed")


def test_a_continued_run_says_so(tmp_path: Path) -> None:
    sim = Simulator(_net(tmp_path, "ab.net", AB))
    first = sim.run(t_span=(0, 1), n_points=3)
    with pytest.warns(bngsim.LiveStartWarning):
        second = sim.run(t_span=(0, 1), n_points=3)
    assert first.notices == []
    assert _codes(second) == ["start.continues_last_run"]
    assert second.notices[0]["category"] == "LiveStartWarning"
    # run_until's own warning, given before the run it makes, is on that run.
    other = Simulator(sim.model)
    with pytest.warns(bngsim.LiveStartWarning):
        step = other.run_until(0.5)
    assert _codes(step) == ["start.continues_last_run"]


def test_a_warning_about_every_row_is_on_every_row_and_once_on_the_stack(
    tmp_path: Path,
) -> None:
    model = _net(tmp_path, "ab.net", AB)
    model.set_concentration("A()", 3.0)
    sim = Simulator(model)
    params = [{"kf": 0.1}, {"kf": 0.2}]
    with pytest.warns(UserWarning, match="run_batch starts every point"):
        rows = sim.run_batch(t_span=(0, 1), n_points=2, params=params)
    assert [_codes(r) for r in rows] == [["start.rows_from_initial"]] * 2
    with pytest.warns(UserWarning, match="run_batch starts every point"):
        stack = sim.run_batch(t_span=(0, 1), n_points=2, params=params, squeeze=True)
    assert _codes(stack) == ["start.rows_from_initial"]


@pytest.mark.parametrize("num_processors", [None, 2])
def test_a_rows_own_notice_is_on_that_row_alone(tmp_path: Path, num_processors) -> None:
    """A row that rounds a count says so, and its neighbour, which rounds none,
    does not: on the caller's thread and on worker threads alike."""
    sim = Simulator(_net(tmp_path, "ab.net", AB), method="ssa")
    with pytest.warns(bngsim.SsaRoundingWarning):
        rows = sim.run_batch(
            t_span=(0, 1),
            n_points=2,
            params=[{"A0": 10.5}, {"A0": 10.0}, {"A0": 7.5}],
            seed=1,
            num_processors=num_processors,
        )
    assert [_codes(r) for r in rows] == [["ssa.rounded_counts"], [], ["ssa.rounded_counts"]]
    stack = Result.squeeze(rows)
    assert _codes(stack) == ["ssa.rounded_counts"]


def test_a_replicate_and_a_scan_point_carry_their_calls_notices(tmp_path: Path) -> None:
    model = _net(tmp_path, "ab.net", AB)
    Simulator(model).run(t_span=(0, 1), n_points=2)
    ssa = Simulator(model, method="ssa")
    with pytest.warns(bngsim.LiveStartWarning), pytest.warns(bngsim.SsaRoundingWarning):
        reps = ssa.run_replicates(2, t_span=(0, 1), n_points=2, seed=3)
    # Each replicate's own (the run left fractions, which it rounds), then the
    # call's, given once before the replicates ran.
    assert [_codes(r) for r in reps] == [["ssa.rounded_counts", "start.continues_last_run"]] * 2
    source = Simulator(_net(tmp_path, "source.net", SOURCE))
    with pytest.warns(bngsim.AtolScaleWarning):
        points = source.parameter_scan("k", [1e-12, 2e-12], t_span=(0, 1), n_points=2)
    assert [_codes(p) for p in points] == [["atol.below_scale"]] * 2


def test_a_warning_given_once_a_process_is_on_every_run(tmp_path: Path, monkeypatch) -> None:
    """The dense-solver warning is given once a process (GH #209); its notice is
    on each run it concerns."""
    monkeypatch.setattr(simulator_module, "_HAS_KLU", False)
    monkeypatch.setattr(simulator_module, "_DENSE_FALLBACK_WARN_NSPECIES", 1)
    monkeypatch.setattr(simulator_module, "_dense_fallback_warned", False)
    sim = Simulator(_net(tmp_path, "ab.net", AB))
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        first = sim.run(t_span=(0, 1), n_points=2, start="initial")
        second = sim.run(t_span=(0, 1), n_points=2, start="initial")
    assert [w.category for w in shown] == [bngsim.DenseSolverFallbackWarning]
    assert _codes(first) == _codes(second) == ["linear_solver.dense_without_klu"]


def test_notices_survive_a_stop_condition_cut(tmp_path: Path) -> None:
    sim = Simulator(_net(tmp_path, "source.net", SOURCE))
    sim.add_stop_condition("X > 4e-13", label="half")
    with pytest.raises(bngsim.StopConditionMet) as stop, pytest.warns(bngsim.AtolScaleWarning):
        sim.run(t_span=(0, 1), n_points=3)
    assert _codes(stop.value.result) == ["atol.below_scale"]


def test_each_notice_is_three_strings_and_a_code_of_the_registry(tmp_path: Path) -> None:
    sim = Simulator(_net(tmp_path, "source.net", SOURCE))
    with pytest.warns(bngsim.AtolScaleWarning):
        first = sim.run(t_span=(0, 1), n_points=2)
    with pytest.warns(bngsim.LiveStartWarning):
        second = sim.run(t_span=(0, 1), n_points=2)
    assert _codes(first) == ["atol.below_scale"]
    assert _codes(second) == ["start.continues_last_run"]
    for r in (first, second):
        for n in r.notices:
            assert set(n) == {"category", "code", "message"}
            assert all(isinstance(v, str) for v in n.values())
            assert n["code"] in _notices.CODES
        assert r.record["notices"] == r.notices


# ─── Review round 1 ────────────────────────────────────────────────────


def test_a_rows_tolerance_notice_is_on_that_row_alone(tmp_path: Path) -> None:
    """The batch warns once, naming every row's species; the notice goes to the
    rows whose species stayed below the tolerance."""
    sim = Simulator(_net(tmp_path, "source.net", SOURCE))
    with pytest.warns(bngsim.AtolScaleWarning):
        rows = sim.run_batch(t_span=(0, 1), n_points=3, params=[{"k": 1e-12}, {"k": 1.0}])
    assert [_codes(r) for r in rows] == [["atol.below_scale"], []]
    with pytest.warns(bngsim.AtolScaleWarning):
        points = sim.parameter_scan("k", [1e-12, 1.0], t_span=(0, 1), n_points=3)
    assert [_codes(p) for p in points] == [["atol.below_scale"], []]


def test_a_logged_notice_names_the_line_that_logged_it(caplog) -> None:
    sim = Simulator(Model.from_sbml_string(SINK))
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        sim.run(t_span=(0, 40), n_points=5)
    (record,) = [
        r
        for r in caplog.records
        if r.getMessage().startswith("analytical Jacobian: CVODE integration failed")
    ]
    assert (record.filename, record.funcName) == ("_simulator.py", "_ode_attempts")


def test_warnings_name_the_lines_they_named_before(tmp_path: Path) -> None:
    """Wrapping a call to collect its notices moves no warning: a warning
    raised with a fixed stacklevel inside run() still names the caller's line,
    and inside a batch row the line in bngsim it named before."""
    sim = Simulator(_net(tmp_path, "ab.net", AB), method="ssa")
    sim.model.set_param("A0", 10.5)
    with pytest.warns(bngsim.SsaRoundingWarning) as shown:
        sim.run(t_span=(0, 1), n_points=2, seed=1, start="initial")
    assert [w.filename for w in shown if w.category is bngsim.SsaRoundingWarning] == [__file__]
    sim.model.reset()
    with pytest.warns(bngsim.SsaRoundingWarning) as shown:
        sim.run_batch(t_span=(0, 1), n_points=2, params=[{"kf": 0.1}], seed=1)
    rounding = [w for w in shown if w.category is bngsim.SsaRoundingWarning]
    assert [Path(w.filename).name for w in rounding] == ["_simulator.py"]


def test_a_run_after_a_retry_says_why_it_used_a_difference_quotient() -> None:
    sim = Simulator(Model.from_sbml_string(SINK))
    first = sim.run(t_span=(0, 40), n_points=5)
    second = sim.run(t_span=(0, 40), n_points=5, start="initial")
    assert _codes(first) == ["jacobian.difference_quotient_retry"]
    assert _codes(second) == ["jacobian.difference_quotient_kept"]
    assert second.notices[0]["category"] == "notice"


def test_a_protocol_run_that_reached_no_steady_state_says_so(tmp_path: Path) -> None:
    from bngsim.convert import parse_bngl_protocol

    model = _net(tmp_path, "growth.net", SOURCE.replace("1 k 1e-12", "1 k 1"))
    spec = parse_bngl_protocol(
        'begin actions\nsimulate({method=>"ode",t_end=>1,n_steps=>2,steady_state=>1})\n'
        "end actions\n"
    )
    with pytest.warns(bngsim.ConversionWarning, match="did not reach a steady state"):
        (run,) = bngsim.run_protocol(model, spec)
    assert _codes(run.result) == ["protocol.steady_state_not_reached"]
    assert run.result.record["run"]["steady_state"] is True


def test_compute_all_sensitivities_keeps_its_notices(tmp_path: Path) -> None:
    import shutil

    if not (shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")):
        pytest.skip("no C compiler for sensitivities")
    derived = AB.replace("    2 A0 10\n", "    2 A0 10\n    3 twice 2*kf\n")
    sim = Simulator(_net(tmp_path, "derived.net", derived))
    with pytest.warns(UserWarning, match="skipping 1 derived"):
        r = sim.compute_all_sensitivities(t_span=(0, 1), n_points=3)
    assert "sens.parameters_skipped" in _codes(r)


def test_a_threads_pending_switch_notices_are_its_own() -> None:
    """A batch row places its stops on a worker thread: what one row's scan
    found and has not said yet is not another's to say."""
    import threading

    from bngsim import _switch_sensitivity as sw

    sw.say_what_was_read(False)
    sw._say_once("t > 1", "issue-1703-test", "a condition %s", "here")
    seen: list[list] = []
    other = threading.Thread(target=lambda: seen.append(list(sw._pending())))
    other.start()
    other.join()
    assert seen == [[]]
    assert sw._pending() == [("t > 1", "issue-1703-test", "a condition %s", ("here",))]
    sw.say_what_was_read(False)
    assert sw._pending() == []


@pytest.mark.parametrize("n_workers", [1, 2])
def test_a_sensitivity_chunks_notices_reach_the_stitched_result(
    tmp_path: Path, monkeypatch, n_workers
) -> None:
    """compute_all_sensitivities runs its chunks on worker threads, which have
    no collector of the call's: each chunk collects its own."""
    import shutil

    if not (shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")):
        pytest.skip("no C compiler for sensitivities")
    real = Simulator._run_sensitivity_chunk

    def noisy(self, params, *args, **kwargs):
        _notices.note("log", "sens.chunk_split", f"a chunk of {list(params)}")
        return real(self, params, *args, **kwargs)

    monkeypatch.setattr(Simulator, "_run_sensitivity_chunk", noisy)
    sim = Simulator(_net(tmp_path, "ab.net", AB))
    r = sim.compute_all_sensitivities(
        t_span=(0, 1), n_points=3, params=["kf", "A0"], chunk_size=1, n_workers=n_workers
    )
    assert sorted(n["message"] for n in r.notices) == ["a chunk of ['A0']", "a chunk of ['kf']"]


@pytest.mark.parametrize("n_workers", [1, 2])
def test_a_failed_chunks_notices_reach_the_recovered_result(
    tmp_path: Path, monkeypatch, n_workers
) -> None:
    """A chunk that fails is retried a column at a time. What it noted before
    it failed rides on its error (issue #1857) and lands on the stitched
    result, not with the discarded error."""
    import shutil

    if not (shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")):
        pytest.skip("no C compiler for sensitivities")
    real = Simulator._run_sensitivity_chunk

    def fails_together(self, params, *args, **kwargs):
        if len(params) > 1:
            _notices.note("log", "sens.chunk_split", "noted by the failed chunk")
            raise bngsim.SimulationError("the chunk failed together")
        return real(self, params, *args, **kwargs)

    monkeypatch.setattr(Simulator, "_run_sensitivity_chunk", fails_together)
    sim = Simulator(_net(tmp_path, "ab.net", AB))
    r = sim.compute_all_sensitivities(
        t_span=(0, 1), n_points=3, params=["kf", "A0"], chunk_size=2, n_workers=n_workers
    )
    assert "noted by the failed chunk" in [n["message"] for n in r.notices]
