"""Issue #438 — the per-run verdict is published, and the decline reason survives
the codegen cache.

Whether a gradient runs on bngsim's analytic ``∂f/∂p`` or on CVODES' internal
difference quotient is a property of the **(build, model)** pair. ``CVodeSensInit1``
takes one sensitivity-RHS callback for every column, so a single rate law bngsim
cannot differentiate declines the analytic derivative for the whole model, and the
difference quotient that replaces it costs an extra RHS evaluation per column per
step — roughly N times the sensitivity cost on an N-parameter fit. On
``Smith_BMCSystBiol2013`` all 25 columns fell back and every gradient start timed
out. No build-level capability key can answer that question, because two models on
one bngsim get different answers and the same model can flip on a derivation-budget
timeout (GH #90).

bngsim knew both halves of the answer and published neither. The verdict was a
private method; the reason went to the logger and **did not survive the cache**.
Since issue #174 the codegen cache key is structural, so a warm cache resolves the
``.so`` without generating any source — and source generation is where the decline
is derived and reported. The first construction of a declining model said why, the
second said nothing at all, and both were on the same fallback. Since the cache is
on disk, the run that hears nothing is typically the second run: the one made after
the first came back empty.

:class:`TestTheReasonSurvivesTheCache` is that measurement, run as a test.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

import bngsim
import bngsim._codegen as cg
import pytest
from bngsim import cache as cache_mod

pytest.importorskip("sympy")

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")


# ─── fixtures ──────────────────────────────────────────────────────────────
#
# One .net model, three rate laws, chosen for what each does to the analytic
# sensitivity RHS:
#
#   DIFFERENTIABLE — the positive control. Nothing declines, so the artifact
#                    carries the symbol and there is no reason to record.
#   UNDERIVABLE    — abs() is the issue's own reproduction. The analytic RHS is
#                    declined and the difference quotient answers the same smooth
#                    question more slowly: "slow", not "wrong".
#   UNCOMPENSATED  — a comparison outside an if() head. The analytic RHS is
#                    declined AND the difference quotient answers a different
#                    question, because it integrates straight through a crossing
#                    whose time moves and drops the saltation jump (GH #146). That
#                    distinction lives in the reason's CLASS, and a persisted
#                    reason has to carry it back.

_NET = """\
begin parameters
    1 S0      1000  # Constant
    2 I0      1  # Constant
    3 beta    0.002  # Constant
    4 gamma   0.15  # Constant
    5 thresh  40.0  # Constant
end parameters
begin functions
    1 betaI() {law}
end functions
begin species
    1 person(state~S) S0
    2 person(state~I) I0
    3 person(state~R) 0
end species
begin reactions
    1 1 2 betaI #_R1
    2 2 3 gamma #_R2
end reactions
begin groups
    1 S                    1
    2 I                    2
    3 R                    3
end groups
"""

DIFFERENTIABLE = "beta*I"
UNDERIVABLE = "beta*abs(I-thresh)*I"
UNCOMPENSATED = "beta*(I>1)*I"


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch) -> Path:
    """A codegen cache of this test's own, cold on entry.

    Both halves matter: the on-disk directory, so "first construction" really is a
    cold cache whatever else the suite has compiled, and the in-process memo, which
    is a second short-circuit past source generation with the same consequence for
    the decline.
    """
    d = tmp_path / "codegen-cache"
    d.mkdir()
    monkeypatch.setattr(cg, "CACHE_DIR", d)
    monkeypatch.setattr(cg, "_PREPARE_CODEGEN_MEMO", {})
    return d


def _model(tmp_path, law: str, name: str = "m.net"):
    net = tmp_path / name
    net.write_text(_NET.format(law=law))
    return bngsim.Model.from_net(net)


def _sim(tmp_path, law: str, name: str = "m.net"):
    """A sensitivity-bearing Simulator, which is the only kind the question is
    about: since issues #209/#217 a plain build emits no sensitivity RHS at all."""
    return bngsim.Simulator(_model(tmp_path, law, name), method="ode", sensitivity_params=["beta"])


def _notes(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("*.sens.json"))


def _declines(caplog) -> list[str]:
    """Every decline bngsim reported, as a consumer listening on its logger sees
    them — the channel PyBNF ships against today."""
    return [
        r.getMessage() for r in caplog.records if "sensitivity RHS is declined" in r.getMessage()
    ]


# ─── the published verdict ─────────────────────────────────────────────────


@needs_cc
class TestTheVerdictIsPublished:
    """Ask 2: promote what ``_codegen_provides_sens_rhs`` already answers.

    The name is not this codebase's to invent — ``_apply_switch_time_sens`` already
    passes ``has_analytic_sens_rhs=self._codegen_provides_sens_rhs()`` into
    ``compute_switch_time_sens``, and PyBNF's resolver already reads an attribute of
    that name and falls through when it is absent (lanl/PyBNF#610).
    """

    def test_a_differentiable_model_is_on_the_analytic_path(self, tmp_path, isolated_cache):
        sim = _sim(tmp_path, DIFFERENTIABLE)
        assert sim.has_analytic_sens_rhs is True
        assert sim.sens_rhs_decline_reason is None

    def test_an_underivable_rate_law_is_not(self, tmp_path, isolated_cache):
        sim = _sim(tmp_path, UNDERIVABLE)
        assert sim.has_analytic_sens_rhs is False
        assert "abs()" in sim.sens_rhs_decline_reason

    def test_it_is_a_plain_bool_a_consumer_can_act_on(self, tmp_path, isolated_cache):
        """Read as an attribute, not called as a method, and a ``bool`` rather than
        anything truthy: PyBNF's resolver takes this route only for an
        ``isinstance(value, bool)``, and treats anything else as no opinion."""
        sim = _sim(tmp_path, DIFFERENTIABLE)
        published = getattr(sim, "has_analytic_sens_rhs", None)
        assert isinstance(published, bool)
        assert published == sim._codegen_provides_sens_rhs()

    def test_a_plain_build_reports_no_decline(self, tmp_path, isolated_cache):
        """The one reading that has to be documented rather than fixed. Since
        issues #209/#217 an artifact built for a plain solve carries no analytic
        ``∂f/∂p`` because nobody asked for one, which is not a decline — so the
        verdict is False and there is no reason to give."""
        model = _model(tmp_path, DIFFERENTIABLE)
        sim = bngsim.Simulator(model, method="ode", codegen=True)
        assert sim.has_analytic_sens_rhs is False
        assert sim.sens_rhs_decline_reason is None


# ─── the reason survives a cache hit ───────────────────────────────────────


@needs_cc
class TestTheReasonSurvivesTheCache:
    """Ask 1, and the measurement the issue was filed on.

    Before this change the table read: first construction, difference quotient,
    decline logged with its reason; second construction, same difference quotient,
    **nothing logged**. Same model, same fallback, and the second says nothing.
    """

    def test_the_second_construction_gives_the_same_reason(self, tmp_path, isolated_cache):
        first = _sim(tmp_path, UNDERIVABLE)
        assert first.codegen_cache_hit is False, "the first build must be the cold one"
        reason = first.sens_rhs_decline_reason
        assert "abs()" in reason

        second = _sim(tmp_path, UNDERIVABLE, name="again.net")
        assert second.codegen_cache_hit is True, "the second build must hit the cache"
        assert second.has_analytic_sens_rhs is False
        assert second.sens_rhs_decline_reason == reason

    def test_the_warm_build_reports_the_decline_on_the_logger_too(
        self, tmp_path, isolated_cache, caplog
    ):
        """A consumer listening on the ``bngsim`` logger — which is what PyBNF
        ships today — hears a cache hit say exactly what a cold build says. The
        replay routes through the same reporter, so the two lines are identical and
        nothing parsing them has to know which it got.
        """
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            _sim(tmp_path, UNDERIVABLE)
            cold = _declines(caplog)
            caplog.clear()
            _sim(tmp_path, UNDERIVABLE, name="again.net")
            warm = _declines(caplog)
        assert len(cold) == 1
        assert warm == cold

    def test_without_the_note_the_warm_build_has_no_opinion(self, tmp_path, isolated_cache):
        """The control for the two tests above: it is the note doing the work.

        Deleting it reproduces a cache entry built by a bngsim older than this
        change — the verdict still stands, because that is read off the artifact,
        and the reason goes quiet. Quiet is the honest answer there, and the pair
        (False, None) is exactly what tells a consumer it has no reason to report
        rather than a model with nothing to report.
        """
        _sim(tmp_path, UNDERIVABLE)
        for note in _notes(isolated_cache):
            note.unlink()

        warm = _sim(tmp_path, UNDERIVABLE, name="again.net")
        assert warm.codegen_cache_hit is True
        assert warm.has_analytic_sens_rhs is False
        assert warm.sens_rhs_decline_reason is None

    def test_the_in_process_memo_replays_it_as_well(self, tmp_path, isolated_cache):
        """The .net path has a second short-circuit past source generation — the
        process-local memo keyed on the file's mtime — and it skips the same step
        for the same reason. Two constructions from the SAME .net file take it,
        where the two above (different file names, same content) take the on-disk
        cache."""
        first = _sim(tmp_path, UNDERIVABLE)
        assert cg._PREPARE_CODEGEN_MEMO, "the memo is what this test is about"
        second = _sim(tmp_path, UNDERIVABLE)
        assert second.sens_rhs_decline_reason == first.sens_rhs_decline_reason

    def test_a_budget_decline_survives_it_too(self, tmp_path, isolated_cache, monkeypatch):
        """The other producer of a decline, and the one the issue singles out.

        A build-time derivation budget (GH #90) means the answer is not even a pure
        function of (build, model source) — the same model can flip on how long the
        derivation took — which is the sharpest reason the question cannot be
        settled by anything but a per-run read. It reaches the recorder from a
        different site than an underivable rate law does, so it is worth its own
        assertion rather than an appeal to the shared chokepoint.
        """
        monkeypatch.setenv("BNGSIM_SENS_DERIV_BUDGET_S", "1e-9")
        cold = _sim(tmp_path, "beta*I/(1 + I/thresh)")
        assert cold.has_analytic_sens_rhs is False
        assert "budget" in cold.sens_rhs_decline_reason

        warm = _sim(tmp_path, "beta*I/(1 + I/thresh)", name="again.net")
        assert warm.codegen_cache_hit is True
        assert warm.sens_rhs_decline_reason == cold.sens_rhs_decline_reason

    def test_a_model_that_did_not_decline_writes_no_note(self, tmp_path, isolated_cache):
        """The cold path pays one small write, and only for a model that declined.
        Nothing is written for the overwhelming majority of models, which are on the
        analytic path and have nothing to say."""
        _sim(tmp_path, DIFFERENTIABLE)
        assert _notes(isolated_cache) == []

    def test_an_unwritable_cache_still_builds(self, tmp_path, isolated_cache, monkeypatch):
        """Recording the reason is best effort. A read-only artifact directory — a
        pre-warmed one on a cluster (GH #203) — must cost the reason on a later
        cache hit and never the build itself."""

        unwritable = tmp_path / "not-a-directory" / "note.sens.json"
        monkeypatch.setattr(cg, "_sens_decline_note_path", lambda _so: unwritable)
        sim = _sim(tmp_path, UNDERIVABLE)
        assert sim.has_analytic_sens_rhs is False
        # This build derived the reason itself, so it still has it; what is lost is
        # the next build's ability to.
        assert "abs()" in sim.sens_rhs_decline_reason
        assert _notes(isolated_cache) == []


# ─── "wrong" and "slow" are different statements ───────────────────────────


@needs_cc
class TestTheFallbackIsNotAlwaysCorrect:
    """The reason's class is the difference between a gradient that is slower and
    one that is wrong, so persisting the text alone would lose the half that
    matters most. ``UNCOMPENSATED`` branches on a comparison whose crossing time
    moves with the parameters and which nothing brackets: the difference quotient
    integrates the variational equation straight through it and drops the jump
    ``(f⁻−f⁺)·dt*/dθ``, so every column is wrong at and after the crossing.
    """

    def test_the_reason_class_round_trips_through_the_note(self, tmp_path, isolated_cache):
        from bngsim._switch_sensitivity import UncompensatedCrossingReason

        cold = _sim(tmp_path, UNCOMPENSATED)
        assert isinstance(cold.sens_rhs_decline_reason, UncompensatedCrossingReason)

        warm = _sim(tmp_path, UNCOMPENSATED, name="again.net")
        assert warm.codegen_cache_hit is True
        assert isinstance(warm.sens_rhs_decline_reason, UncompensatedCrossingReason)

    def test_a_merely_underivable_law_is_not_tagged(self, tmp_path, isolated_cache):
        """The other direction, without which the check above would pass on a note
        reader that tagged everything."""
        from bngsim._switch_sensitivity import UncompensatedCrossingReason

        warm_source = _sim(tmp_path, UNDERIVABLE)
        assert not isinstance(warm_source.sens_rhs_decline_reason, UncompensatedCrossingReason)
        warm = _sim(tmp_path, UNDERIVABLE, name="again.net")
        assert not isinstance(warm.sens_rhs_decline_reason, UncompensatedCrossingReason)

    def test_the_replayed_warning_still_says_the_fallback_is_wrong(
        self, tmp_path, isolated_cache, caplog
    ):
        """The class decides the sentence the warning ends with, and a cache hit
        must not downgrade it to "correct, but slower"."""
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            _sim(tmp_path, UNCOMPENSATED)
            caplog.clear()
            _sim(tmp_path, UNCOMPENSATED, name="again.net")
        warm = _declines(caplog)
        assert len(warm) == 1
        assert "does NOT recover" in warm[0]


# ─── the note belongs to the cache ─────────────────────────────────────────


class TestTheNoteIsOneOfBngsimsOwnFiles:
    """``bngsim-cache`` classifies every entry by name and never removes one it
    does not recognize, so a file this module writes and that one does not know
    would be reported as somebody else's and kept forever.

    Name-only, like the classifier itself, so these need no compiler.
    """

    def _names(self, cache_dir: Path, hashes=("aaaa1111", "bbbb2222")) -> dict[str, Path]:
        suffix = cg._shared_lib_suffix()
        out: dict[str, Path] = {}
        for h in hashes:
            stem = cg._artifact_stem(h)
            art = cache_dir / f"{stem}{suffix}"
            art.write_bytes(b"x" * 1024)
            note = cg._sens_decline_note_path(art)
            note.write_text('{"version": 1, "class": "plain", "reason": "because"}')
            out[h] = art
        return out

    def test_a_note_is_classified_as_bngsims_own(self, tmp_path):
        note = tmp_path / f"{cg._artifact_stem('abcd1234')}.sens.json"
        note.write_text("{}")
        assert cache_mod.classify(note) == cache_mod.KIND_NOTE
        assert cache_mod.classify(note) != cache_mod.KIND_FOREIGN
        assert cache_mod.artifact_key(note) == cg._CODEGEN_CACHE_KEY

    def test_a_note_is_not_counted_as_an_artifact(self, tmp_path, monkeypatch):
        """It is metadata about an artifact, not one: nothing loads it, and letting
        it into the live/orphaned counts would report one compiled model as two."""
        monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
        self._names(tmp_path, hashes=("aaaa1111",))
        info = cache_mod.codegen_cache_info(tmp_path)
        assert len(info.live) == 1
        assert info.by_kind[cache_mod.KIND_NOTE][0] == 1
        assert info.by_kind[cache_mod.KIND_FOREIGN][0] == 0

    def test_prune_takes_a_note_with_its_artifact_and_leaves_the_others(self, tmp_path):
        """A note is worth nothing without the artifact it describes, and describes
        nothing once that artifact is gone."""
        arts = self._names(tmp_path)
        # Age one pair past the bound and leave the other fresh.
        old = time.time() - 86400
        for p in (arts["aaaa1111"], cg._sens_decline_note_path(arts["aaaa1111"])):
            os.utime(p, (old, old))

        sweep = cache_mod.prune_codegen_cache(tmp_path, older_than="1h", min_age="0s")
        removed = {p.path.name for p in sweep.removed}
        assert removed == {
            arts["aaaa1111"].name,
            cg._sens_decline_note_path(arts["aaaa1111"]).name,
        }
        assert cg._sens_decline_note_path(arts["bbbb2222"]).exists()

    def test_prune_collects_a_note_whose_artifact_is_gone(self, tmp_path):
        """Including one left by a sweep that predates the pairing, and the
        process-unique temporary of a note that never landed."""
        stray = tmp_path / f"{cg._artifact_stem('cccc3333')}.sens.json"
        stray.write_text("{}")
        temp = tmp_path / f"{cg._artifact_stem('dddd4444')}.4711_0.sens.json"
        temp.write_text("{}")

        sweep = cache_mod.prune_codegen_cache(tmp_path, older_than="1h", min_age="0s")
        assert {p.path.name for p in sweep.removed} == {stray.name, temp.name}

    def test_clear_removes_notes(self, tmp_path):
        self._names(tmp_path)
        cache_mod.clear_codegen_cache(tmp_path)
        assert list(tmp_path.iterdir()) == []


# ─── the build that has no artifact to write a note beside ─────────────────


@needs_cc
class TestTheJitSourcePathRecordsItToo:
    """The MIR JIT backend compiles no ``.so``, so there is nothing to put a note
    next to — and it needs none, because it regenerates the source on every build
    and therefore derives the reason every time. What it does need is somewhere to
    put it, which is the model: a second Simulator inheriting that source runs no
    codegen of its own and would otherwise have nothing to read.

    Driven through the ``prepare_*`` entry points directly rather than through a
    Simulator, so this runs on a build without MIR compiled in — which is every
    default build.
    """

    def test_the_model_path_records_the_reason_on_the_model(self, tmp_path, isolated_cache):
        model = _model(tmp_path, UNDERIVABLE)
        model._want_output_sens = True  # what sensitivity_params= sets
        source = cg.prepare_model_codegen_source(model)
        assert "bngsim_codegen_sens_rhs" not in source
        assert "abs()" in model._codegen_sens_decline

    def test_a_model_that_did_not_decline_records_none(self, tmp_path, isolated_cache):
        """Recorded on every prepare, including as ``None``, so a later plain build
        of the same model cannot leave an earlier decline standing."""
        model = _model(tmp_path, UNDERIVABLE)
        model._want_output_sens = True
        cg.prepare_model_codegen_source(model)
        assert model._codegen_sens_decline is not None

        model._want_output_sens = False  # nobody asks, so nothing declines
        cg.prepare_model_codegen_source(model)
        assert model._codegen_sens_decline is None

    def test_the_net_path_records_it_on_the_thread(self, tmp_path, isolated_cache):
        """``prepare_codegen_source`` takes a path rather than a Model, so it
        records to the thread-local that ``carry_codegen_stats`` reads."""
        model = _model(tmp_path, UNDERIVABLE)
        model._want_output_sens = True
        cg.prepare_codegen_source(str(tmp_path / "m.net"), model)
        assert "abs()" in cg.last_sens_rhs_decline()

        cg.carry_codegen_stats(model)
        assert model._codegen_sens_decline == cg.last_sens_rhs_decline()


# ─── the memo follows the artifact ─────────────────────────────────────────


@needs_cc
class TestTheVerdictTracksTheArtifactItIsAbout:
    """Found while publishing the verdict, fixed here because publishing it is what
    makes it reachable.

    ``_codegen_provides_sens_rhs`` reads the artifact once and remembers the answer.
    ``compute_all_sensitivities`` and ``steady_state`` take ``sensitivity_params``
    as a method argument and rebuild the artifact for themselves — a plain build
    carries no ``bngsim_codegen_sens_rhs`` at all since issues #209/#217, so the
    rebuild changes the answer. A memo taken before that swap describes an artifact
    the run no longer installs, and it took publishing the reader for anyone to be
    able to take one at that moment.
    """

    def test_the_answer_changes_when_the_artifact_does(self, tmp_path, isolated_cache):
        sim = bngsim.Simulator(_model(tmp_path, DIFFERENTIABLE), method="ode", codegen=True)
        assert sim.has_analytic_sens_rhs is False, "a plain build, so nobody asked"

        sim.compute_all_sensitivities((0.0, 1.0), 3, params=["beta"])
        assert sim.has_analytic_sens_rhs is True, "the rebuilt artifact carries it"
