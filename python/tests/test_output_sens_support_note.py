"""The output-sensitivity support map is the cached artifact's own (issue #1004).

A sensitivity run attaches ``{function: why its output sensitivity is not
available}`` to its result, and ``output_sensitivities([...])`` raises that
reason for a function the compiled code holds no derivative for. The compiled
code comes from the on-disk cache. The map was worked out again in every
process, by an analysis that runs under a wall-clock budget, so it is not a
function of the model alone: where the build that wrote the artifact was cut
by the budget and a later process's analysis was not, the later map called a
function supported that the artifact holds NaN for, and the selector returned
NaN with nothing said.

The map is now written beside the artifact when it is built, and read back
where the artifact is taken from the cache.

Two processes are two ``Model`` objects here, each with its own analysis, on
one cache directory. The budget's expiry is simulated as the issue's repro
does: its own exception is raised from the second function's derivation on.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

import bngsim
import bngsim._codegen as cg
import numpy as np
import pytest
from bngsim import _jacobian
from bngsim import cache as cache_mod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _codegen_backend import under_jit_no_cc_artifact  # noqa: E402

# Tables added after load here are fixtures that nothing in the model reads,
# which add_table_function warns about (issue #772).
pytestmark = pytest.mark.filterwarnings("ignore:nothing in the model reads table function")

pytest.importorskip("sympy")

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")

_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"
CHAIN_NET = str(DATA_DIR / "expr_sens_chain.net")
FUNCTIONS = ["scaled", "ratio", "combo", "tdep"]


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "codegen-cache"
    d.mkdir()
    monkeypatch.setattr(cg, "CACHE_DIR", d)
    return d


def _run():
    """A sensitivity run of a model loaded for it: a process of its own, as
    far as the analysis goes."""
    sim = bngsim.Simulator(
        bngsim.Model.from_net(CHAIN_NET),
        method="ode",
        codegen=True,
        sensitivity_params=["k1", "scale"],
    )
    return sim, sim.run(t_span=(0.0, 8.0), n_points=5)


@contextlib.contextmanager
def _cut_after_the_first():
    """The budget runs out after the first function, inside the block."""
    real = _jacobian.differentiate_expression_output_partials
    calls = [0]

    def cut(body, **kw):
        calls[0] += 1
        if calls[0] >= 2:
            raise _jacobian._DerivationBudgetExceeded
        return real(body, **kw)

    patch = pytest.MonkeyPatch()
    patch.setattr(_jacobian, "differentiate_expression_output_partials", cut)
    try:
        yield
    finally:
        patch.undo()


def _notes(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("*.outputs.json"))


def _artifacts(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob(f"*{cg._shared_lib_suffix()}"))


# ─── The issue ───────────────────────────────────────────────────────────────


@needs_cc
@under_jit_no_cc_artifact
def test_a_later_process_raises_the_reason_the_build_was_cut_for(isolated_cache):
    """The first run's analysis is cut after ``scaled``; the second is not cut
    and is handed the first one's compiled code. It returned NaN for
    ``ratio``: ``[nan, nan, nan, nan]``."""
    with _cut_after_the_first():
        _sim, first = _run()
        with pytest.raises(Exception, match="'ratio' has no output sensitivity.*budget"):
            first.output_sensitivities(["ratio"])
        cut_scaled = np.asarray(first.output_sensitivities(["scaled"]))
    assert len(_artifacts(isolated_cache)) == 1

    _sim, second = _run()
    assert len(_artifacts(isolated_cache)) == 1  # the first one's, from the cache
    with pytest.raises(Exception, match="'ratio' has no output sensitivity.*budget"):
        second.output_sensitivities(["ratio"])
    np.testing.assert_array_equal(np.asarray(second.output_sensitivities(["scaled"])), cut_scaled)


@needs_cc
def test_a_build_that_was_not_cut_is_read_as_supported(isolated_cache):
    """Control. Nothing is cut in either run, and every function is returned
    in both."""
    _sim, first = _run()
    _sim, second = _run()
    for name in FUNCTIONS:
        np.testing.assert_array_equal(
            np.asarray(second.output_sensitivities([name])),
            np.asarray(first.output_sensitivities([name])),
        )
        assert np.all(np.isfinite(np.asarray(second.output_sensitivities([name]))))


@needs_cc
@under_jit_no_cc_artifact
def test_the_other_way_round_the_artifact_that_was_not_cut_is_not_refused(isolated_cache):
    """The first run is not cut and writes the artifact; the second one's
    analysis would be cut. It is handed code that holds every derivative, and
    was told ``ratio`` has none: a refusal of a number that is there."""
    _sim, first = _run()
    want = np.asarray(first.output_sensitivities(["ratio"]))
    with _cut_after_the_first():
        _sim, second = _run()
        np.testing.assert_array_equal(np.asarray(second.output_sensitivities(["ratio"])), want)


@needs_cc
@under_jit_no_cc_artifact
def test_a_build_that_finds_the_artifact_installed_leaves_its_note(isolated_cache, monkeypatch):
    """Two processes can miss the cache together. One that reaches the
    compiler after the other has installed the artifact is handed that one,
    with whatever its own budget did: the note beside it is the first one's,
    and stays. (Two that are both compiling each install their own: see the
    pull request's "Left alone".)"""
    _sim, first = _run()
    want = np.asarray(first.output_sensitivities(["ratio"]))
    (note,) = _notes(isolated_cache)
    before = note.read_text()

    real = cg.get_cached_so
    looked = [0]

    def missed_once(model_hash):
        looked[0] += 1
        return None if looked[0] == 1 else real(model_hash)

    monkeypatch.setattr(cg, "get_cached_so", missed_once)
    with _cut_after_the_first():
        _sim, second = _run()
        assert looked[0] >= 1
        assert note.read_text() == before
        np.testing.assert_array_equal(np.asarray(second.output_sensitivities(["ratio"])), want)


@needs_cc
@under_jit_no_cc_artifact
def test_the_model_that_built_the_artifact_answers_for_it_itself(isolated_cache):
    """Two processes that both compile each install their own artifact, and
    the note beside the one that lands can be the other's. The model whose
    analysis the code was emitted from does not ask the note: with a note
    beside it that calls every function supported, as a build that was not
    cut writes, it raises its own reason, as it did before there was a note.
    Reading the note here returned NaN in 3 of 8 concurrent cold starts, and
    reading it once the file had been replaced in 17 of 24: the process has
    loaded its own build by then."""
    model = bngsim.Model.from_net(CHAIN_NET)

    def run():
        sim = bngsim.Simulator(
            model, method="ode", codegen=True, sensitivity_params=["k1", "scale"]
        )
        return sim.run(t_span=(0.0, 8.0), n_points=5)

    with _cut_after_the_first():
        first = run()
    with pytest.raises(Exception, match="'ratio' has no output sensitivity.*budget"):
        first.output_sensitivities(["ratio"])
    (artifact,) = _artifacts(isolated_cache)
    cg.write_output_sens_support_note(
        artifact, dict.fromkeys(FUNCTIONS), cg._output_sens_note_key(model)
    )
    model.reset()
    second = run()
    with pytest.raises(Exception, match="'ratio' has no output sensitivity.*budget"):
        second.output_sensitivities(["ratio"])


@needs_cc
@under_jit_no_cc_artifact
def test_the_registry_does_not_keep_an_analysis_alive(isolated_cache, caplog):
    """What the process remembers of an artifact it built is a weak
    reference: the analysis goes with its models. (A DEBUG record of the run
    names its model, and a captured record keeps it alive for the length of
    the test: the logger is held at WARNING, wherever an earlier test left
    it.)"""
    import gc
    import logging

    caplog.set_level(logging.WARNING, logger="bngsim")
    _run()
    (artifact,) = _artifacts(isolated_cache)
    gc.collect()
    assert cg._output_sens_emitted[str(artifact)]() is None


# ─── A model that is not keyed as the builder was ────────────────────────────

RUN = dict(method="ode", sensitivity_params=["k1", "scale"])
TABLE = dict(times=[0.0, 10.0], values=[1.0, 3.0])


def _refused_a_table_after_a_run(model, name="ratio"):
    """*model* is run, refused a table function named for one of its
    functions (issue #772), and run again: both results and both artifacts.
    (At the default ``codegen``, where a model that has run keeps its
    artifact.)"""
    first_sim = bngsim.Simulator(model, **RUN)
    first = first_sim.run(t_span=(0.0, 8.0), n_points=5)
    model.reset()
    with pytest.raises(bngsim.ModelError, match=f"already has a function named '{name}'"):
        model.add_table_function(name, **TABLE)
    second_sim = bngsim.Simulator(model, **RUN)
    second = second_sim.run(t_span=(0.0, 8.0), n_points=5)
    return first, second, first_sim._codegen_so_path, second_sim._codegen_so_path


@needs_cc
@under_jit_no_cc_artifact
def test_a_model_refused_a_table_function_keeps_what_the_note_says(isolated_cache):
    """A table function with a function's name took that function's place in
    the compiled code, where it has no derivative (issue #979), while the
    interpreter kept the function's body. It is refused now (issue #772). The
    model here was handed the artifact from the cache and told its map by the
    note; refused the table, it is the model the note is of, and ``ratio``
    and ``scaled`` are both returned as before."""
    _sim, built = _run()
    (artifact,) = _artifacts(isolated_cache)
    assert cg.read_output_sens_support_note(artifact, FUNCTIONS)["ratio"] is None

    first, second, so_first, so_second = _refused_a_table_after_a_run(
        bngsim.Model.from_net(CHAIN_NET)
    )
    assert Path(so_first) == Path(so_second) == artifact
    for name in ("ratio", "scaled"):
        want = np.asarray(built.output_sensitivities([name]))
        np.testing.assert_array_equal(np.asarray(first.output_sensitivities([name])), want)
        np.testing.assert_array_equal(np.asarray(second.output_sensitivities([name])), want)


@needs_cc
def test_the_builder_refused_a_table_function_answers_from_what_it_built(isolated_cache):
    """Control. The same for the model that built the artifact: the analysis
    its code was emitted from is still this model's."""
    first, second, so_first, so_second = _refused_a_table_after_a_run(
        bngsim.Model.from_net(CHAIN_NET)
    )
    assert Path(so_first) == Path(so_second)
    want = np.asarray(first.output_sensitivities(["ratio"]))
    assert np.all(np.isfinite(want))
    np.testing.assert_array_equal(np.asarray(second.output_sensitivities(["ratio"])), want)


@needs_cc
@pytest.mark.parametrize("who", ["the builder", "a model handed the artifact"])
@under_jit_no_cc_artifact
def test_what_a_cut_build_has_no_derivative_for_holds_under_another_key(isolated_cache, who):
    """The build was cut after ``scaled``. The model is then given a table
    function that is named for none of its functions, which is enough for it
    to be keyed otherwise, and it keeps its artifact. Its own analysis, not
    cut, calls ``ratio`` supported, and the compiled code holds NaN for it:
    with the note read as no note at all, ``[nan, nan, nan, nan]`` came
    back, which is the issue again. The build's reason is raised."""
    with _cut_after_the_first():
        model = bngsim.Model.from_net(CHAIN_NET)
        built_sim = bngsim.Simulator(model, **RUN)
        built = built_sim.run(t_span=(0.0, 8.0), n_points=5)
    want = np.asarray(built.output_sensitivities(["scaled"]))
    if who != "the builder":
        model = bngsim.Model.from_net(CHAIN_NET)
        handed = bngsim.Simulator(model, **RUN).run(t_span=(0.0, 8.0), n_points=5)
        with pytest.raises(Exception, match="'ratio' has no output sensitivity.*budget"):
            handed.output_sensitivities(["ratio"])
    model.reset()
    model.add_table_function("a_table_of_its_own", **TABLE)
    sim = bngsim.Simulator(model, **RUN)
    result = sim.run(t_span=(0.0, 8.0), n_points=5)
    assert Path(sim._codegen_so_path) == Path(built_sim._codegen_so_path)
    with pytest.raises(Exception, match="'ratio' has no output sensitivity.*budget"):
        result.output_sensitivities(["ratio"])
    np.testing.assert_array_equal(np.asarray(result.output_sensitivities(["scaled"])), want)


def _handed_a_cut_artifact():
    """A model that is handed the artifact of a cut build and runs it."""
    with _cut_after_the_first():
        _run()
    model = bngsim.Model.from_net(CHAIN_NET)
    bngsim.Simulator(model, **RUN).run(t_span=(0.0, 8.0), n_points=5)
    model.reset()
    return model


def _handed_a_cut_artifact_then_given_a_table_over(name):
    """The same model, then given a table function."""
    model = _handed_a_cut_artifact()
    model.add_table_function(name, **TABLE)
    return model


@needs_cc
@pytest.mark.parametrize("over", ["scaled", "ratio"])
@under_jit_no_cc_artifact
def test_a_table_over_a_function_of_a_cut_build_is_refused(isolated_cache, over):
    """The table is over ``scaled``, which the build has a derivative for, or
    over ``ratio``, which it was cut before. It is refused (issue #772), and
    for the model refused it the build's reasons stand: ``scaled`` is
    returned, and ``ratio`` is refused for the budget."""
    model = _handed_a_cut_artifact()
    with pytest.raises(bngsim.ModelError, match=f"already has a function named '{over}'"):
        model.add_table_function(over, **TABLE)
    result = bngsim.Simulator(model, **RUN).run(t_span=(0.0, 8.0), n_points=5)
    assert np.all(np.isfinite(np.asarray(result.output_sensitivities(["scaled"]))))
    with pytest.raises(Exception, match="'ratio' has no output sensitivity.*budget"):
        result.output_sensitivities(["ratio"])


@needs_cc
@under_jit_no_cc_artifact
def test_the_builds_reasons_hold_where_the_model_cannot_be_analysed(isolated_cache, monkeypatch):
    """The same model, and its own analysis fails: what the note says the
    build has no derivative for is what there is to go by."""
    model = _handed_a_cut_artifact_then_given_a_table_over("a_table_of_its_own")

    def fails(_model):
        raise RuntimeError("no analysis to be had")

    monkeypatch.setattr(cg, "output_sens_support", fails)
    result = bngsim.Simulator(model, **RUN).run(t_span=(0.0, 8.0), n_points=5)
    with pytest.raises(Exception, match="'ratio' has no output sensitivity.*budget"):
        result.output_sensitivities(["ratio"])
    assert np.all(np.isfinite(np.asarray(result.output_sensitivities(["scaled"]))))


@needs_cc
@under_jit_no_cc_artifact
def test_a_note_is_read_by_its_builders_key(isolated_cache):
    """The note names what its builder's analysis was keyed by, and reads as
    no note under any other key."""
    sim, _result = _run()
    (artifact,) = _artifacts(isolated_cache)
    key = cg._output_sens_note_key(sim._model)
    assert json.loads(_notes(isolated_cache)[0].read_text())["key"] == key
    assert cg.read_output_sens_support_note(artifact, FUNCTIONS, key) is not None
    assert cg.read_output_sens_support_note(artifact, FUNCTIONS, "another") is None
    tabled = bngsim.Model.from_net(CHAIN_NET)
    tabled.add_table_function("a_table_of_its_own", **TABLE)
    assert cg._output_sens_note_key(tabled) != key
    assert cg._output_sens_note_key(bngsim.Model.from_net(CHAIN_NET)) == key


@needs_cc
@under_jit_no_cc_artifact
def test_a_note_that_cannot_be_installed_does_not_fail_the_build(isolated_cache):
    """Something that is no file sits where the note goes: the artifact is
    built, the run returns, and the map is worked out from the model."""
    _sim, first = _run()
    want = np.asarray(first.output_sensitivities(["ratio"]))
    (artifact,) = _artifacts(isolated_cache)
    (note,) = _notes(isolated_cache)
    artifact.unlink()
    note.unlink()
    note.mkdir()
    _sim, second = _run()
    assert artifact.exists() and note.is_dir()
    np.testing.assert_array_equal(np.asarray(second.output_sensitivities(["ratio"])), want)


# ─── The note ────────────────────────────────────────────────────────────────


@needs_cc
@under_jit_no_cc_artifact
def test_the_note_is_written_beside_the_artifact_it_is_of(isolated_cache):
    with _cut_after_the_first():
        sim, _result = _run()
    (artifact,) = _artifacts(isolated_cache)
    (note,) = _notes(isolated_cache)
    assert note == cg._output_sens_note_path(artifact)
    assert Path(sim._codegen_so_path) == artifact
    support = cg.read_output_sens_support_note(artifact, FUNCTIONS)
    assert list(support) == FUNCTIONS
    assert support["scaled"] is None
    assert all("budget" in support[name] for name in FUNCTIONS[1:])


@needs_cc
@under_jit_no_cc_artifact
def test_a_process_that_is_handed_the_artifact_does_not_analyse_the_model(
    isolated_cache, monkeypatch
):
    """The map is read. The analysis took 3 s for a model of 301 functions and
    21 s for one of 501, in the first sensitivity run of every process, for an
    error message."""
    _run()
    made = [0]
    real = cg._compute_output_sens_analysis

    def counted(*args, **kwargs):
        made[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(cg, "_compute_output_sens_analysis", counted)
    sim, result = _run()
    assert sim._expression_sens_support() == dict.fromkeys(FUNCTIONS)
    assert np.all(np.isfinite(np.asarray(result.output_sensitivities(["ratio"]))))
    assert made[0] == 0


@needs_cc
@pytest.mark.parametrize(
    "damage",
    [
        "gone",
        "not json",
        "another version",
        "other functions",
        "another order",
        "a reason that is no text",
        "a name twice",
    ],
)
@under_jit_no_cc_artifact
def test_an_artifact_with_no_note_to_read_is_analysed_as_before(
    isolated_cache, monkeypatch, damage
):
    """An artifact written before the note existed has none, and one whose
    note is damaged, of another version, or not of this model's functions is
    as good as that: the map is worked out from the model, once."""
    _run()
    (note,) = _notes(isolated_cache)
    payload = json.loads(note.read_text())
    if damage == "gone":
        note.unlink()
    elif damage == "not json":
        note.write_text("{")
    elif damage == "another version":
        note.write_text(json.dumps(payload | {"version": 99}))
    elif damage == "other functions":
        payload["functions"][1][0] = "other"
        note.write_text(json.dumps(payload))
    elif damage == "another order":
        payload["functions"].reverse()
        note.write_text(json.dumps(payload))
    elif damage == "a reason that is no text":
        payload["functions"][1][1] = 3
        note.write_text(json.dumps(payload))
    else:
        payload["functions"][1][0] = payload["functions"][0][0]
        note.write_text(json.dumps(payload))

    made = [0]
    real = cg._compute_output_sens_analysis

    def counted(*args, **kwargs):
        made[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(cg, "_compute_output_sens_analysis", counted)
    sim, _result = _run()
    assert sim._expression_sens_support() == dict.fromkeys(FUNCTIONS)
    assert made[0] == 1


def test_a_note_that_cannot_be_written_does_not_fail_the_build(tmp_path):
    """A read-only or full cache directory loses the note and nothing
    else."""
    unwritable = tmp_path / "not-a-directory" / "rhs_x.so"
    assert cg.write_output_sens_support_note(unwritable, {"f": None}) is None
    assert cg.read_output_sens_support_note(unwritable) is None


def test_what_is_written_is_what_is_read(tmp_path):
    artifact = tmp_path / "rhs_abc.so"
    support = {"f": None, "g": "no derivative: it reads a table function", "h": None}
    note = cg.write_output_sens_support_note(artifact, support)
    assert note == tmp_path / "rhs_abc.outputs.json"
    assert cg.read_output_sens_support_note(artifact) == support
    assert list(cg.read_output_sens_support_note(artifact)) == ["f", "g", "h"]
    assert cg.read_output_sens_support_note(artifact, ["f", "g", "h"]) == support
    assert cg.read_output_sens_support_note(artifact, ["f", "g"]) is None
    assert list(tmp_path.iterdir()) == [note]  # no temporary left behind


# ─── The cache's housekeeping ────────────────────────────────────────────────


def _pair(cache_dir: Path, model_hash: str) -> tuple[Path, Path]:
    artifact = cache_dir / f"{cg._artifact_stem(model_hash)}{cg._shared_lib_suffix()}"
    artifact.write_bytes(b"x" * 1024)
    note = artifact.with_suffix(".outputs.json")
    note.write_text('{"version": 1, "functions": []}')
    return artifact, note


def test_the_note_is_bngsims_own_and_no_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
    _artifact, note = _pair(tmp_path, "aaaa1111")
    assert cache_mod.classify(note) == cache_mod.KIND_NOTE
    assert cache_mod.artifact_key(note) == cg._CODEGEN_CACHE_KEY
    info = cache_mod.codegen_cache_info(tmp_path)
    assert len(info.live) == 1
    assert info.by_kind[cache_mod.KIND_NOTE][0] == 1
    assert info.by_kind[cache_mod.KIND_FOREIGN][0] == 0


def test_prune_takes_the_note_with_its_artifact_and_leaves_the_others(tmp_path):
    import time

    old_artifact, old_note = _pair(tmp_path, "aaaa1111")
    _kept_artifact, kept_note = _pair(tmp_path, "bbbb2222")
    stray = tmp_path / f"{cg._artifact_stem('cccc3333')}.outputs.json"
    stray.write_text("{}")
    old = time.time() - 86400
    for path in (old_artifact, old_note):
        os.utime(path, (old, old))
    sweep = cache_mod.prune_codegen_cache(tmp_path, older_than="1h", min_age="0s")
    assert {p.path.name for p in sweep.removed} == {old_artifact.name, old_note.name, stray.name}
    assert kept_note.exists()


def test_clear_removes_the_note(tmp_path):
    _pair(tmp_path, "aaaa1111")
    cache_mod.clear_codegen_cache(tmp_path)
    assert list(tmp_path.iterdir()) == []
