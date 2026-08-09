"""Issues #209/#217 — a sensitivity RHS is emitted only when a sensitivity is asked for.

``generate_combined_from_model`` called ``generate_sens_from_model`` unconditionally,
so ``Simulator(model, method="ode")`` with no ``sensitivity_params`` derived the
Functional sensitivity RHS with sympy, emitted it, and compiled it into the cached
``.so`` — for a run that never calls ``CVodeSensInit1``. On ``BIOMD0000000496``
(295 species, 333 functional reactions, cold cache) that was 39.5 s of construction
and a 26.7 MB ``.so``, against 21.5 s and 1.8 MB once the derivation is skipped.
The GH #198 output-sensitivity evaluator three lines below in the same function was
already gated on ``_want_output_sens`` for exactly this reason; ``∂f/∂p`` is the
more expensive of the two and was the one left ungated.

**#209 gated only the Functional/MM half, and #217 is why that was the wrong
place to stop.** The reasoning for leaving the Elementary half unconditional was
that it is plain text emission with no sympy in it, so gating it would buy no
*derivation* time — only source size — while costing every all-Elementary model
its byte-identical source. Correct about the derivation, wrong about the size: on
the 20 largest ``.net`` models the Elementary sens RHS is **55.6% of the plain
build's C source**, and because ``_resolve_opt_flag`` picks its tier from total
translation-unit size, that dead weight held five of them — ``fceri_fyn`` among
them — at a *lower* ``-O`` for the RHS the solve actually calls.

The byte-identical-source cost never existed either. A plain build and a
sensitivity build have not shared a cache entry since #177 put ``:sens_term_scale``
in the key, so the widening changes what is *in* the plain entry, not how many
entries there are — see :meth:`TestTheCacheKeyCarriesTheFlag.
test_plain_and_sensitivity_were_already_separate_entries`.

The gate is cheap. **Not** silently downgrading a sensitivity run to CVODES'
difference quotient is the whole cost of it, and it takes four things, each with a
test below:

* the resolved flag reaches BOTH cache keys, so a plain-run ``.so`` is never served
  to a sensitivity run with the symbol simply absent (the issue #51 inertness
  trap). Measured honestly, that one is belt-and-braces today: ``emit_output_sens``
  and #177's ``:sens_term_scale`` already separate the two keys, so the mutation
  that blinds the keys to the new flag only trips
  :meth:`test_the_structural_key_moves_with_the_flag` — the two "compiles two
  distinct artifacts" tests below pass for that older reason. They earn their place
  by pinning the *wiring* instead: reverting either production entry point to ask
  unconditionally fails them;
* an entry point that takes ``sensitivity_params`` as a *method* argument
  regenerates — ``compute_all_sensitivities`` and ``steady_state``, whose shared
  helper used to gate its regeneration on ``n_functions > 0`` and would therefore
  have kept a plain artifact for every Michaelis-Menten model;
* the ``.net`` path, whose model-based GH #67 hook emits the same symbol;
* and the constructor's artifact-reuse block, which handed a sensitivity Simulator
  the plain ``.so`` an earlier Simulator had left on the model. That one was
  already dropping ``bngsim_codegen_output_sens`` before this issue.

``sens_dfdp_source`` is the assertion to reach for wherever a steady state is in
play: it reports which path the ∂f/∂p factor actually took, so "codegen" is a
statement about the artifact that was really loaded rather than about the source
that was really generated.
"""

from __future__ import annotations

import ctypes

import bngsim
import pytest
from bngsim import _codegen as cg

pytest.importorskip("sympy")


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")


# ─── fixtures ──────────────────────────────────────────────────────────────
#
# FUNCTIONAL is the class the gate started FOR (its ∂f/∂p is the sympy derivation);
# MM_NO_FUNCTIONS is the class the gate's plumbing is most likely to drop, because
# a Michaelis-Menten model needs no functions at all and the regeneration trigger
# used to key on ``n_functions > 0``; ELEMENTARY is the class #209 exempted and
# #217 brought in — its sensitivity RHS costs no derivation time, which is why it
# looked free to leave alone, and is over half the emitted source on a large model,
# which is why it was not.

FUNCTIONAL = """\
begin parameters
    1 kmax   3.5  # Constant
    2 Km     4.0  # Constant
    3 kdeg   0.2  # Constant
end parameters
begin functions
    1 vsat() kmax*Atot/(Km + Atot)
end functions
begin species
    1 A() 6.0
    2 B() 1.0
end species
begin reactions
    1 1 2 vsat #_R1
    2 2 1 kdeg #_R2
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
end groups
"""

MM_NO_FUNCTIONS = """\
begin parameters
    1 kcat  2.0  # Constant
    2 Km    35.0  # Constant
    3 kdeg  0.05  # Constant
end parameters
begin species
    1 S() 120
    2 E() 25
    3 P() 0
end species
begin reactions
    1 1,2 3,2 MM kcat Km #_R1
    2 3 0 kdeg #_R2
end reactions
begin groups
    1 St                   1
    2 Et                   2
    3 Pt                   3
end groups
"""

ELEMENTARY = """\
begin parameters
    1 k1     0.3  # Constant
    2 k2     0.1  # Constant
end parameters
begin species
    1 A() 10.0
    2 B() 0.0
end species
begin reactions
    1 1 2 k1 #_R1
    2 2 1 k2 #_R2
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
end groups
"""


def _model(tmp_path, text, name="m.net"):
    net = tmp_path / name
    net.write_text(text)
    return bngsim.Model.from_net(net)


def _sbml_functional(tmp_path):
    """The model-based codegen path (``prepare_model_codegen``), which a .net
    model never takes — it carries ``_net_path`` and goes through
    ``prepare_codegen`` instead."""
    pytest.importorskip("antimony")
    return bngsim.Model.from_antimony_string(
        "model mm; S=10; P=0; Vmax=1.4; Km=2.5; J0: S -> P; Vmax*S/(Km + S); end"
    )


def _emits_sens_rhs(source_or_so) -> bool:
    if isinstance(source_or_so, str) and "bngsim_codegen_rhs" in source_or_so:
        return "bngsim_codegen_sens_rhs" in source_or_so
    return hasattr(ctypes.CDLL(str(source_or_so)), "bngsim_codegen_sens_rhs")


# ─── the gate ──────────────────────────────────────────────────────────────


class TestTheGate:
    @pytest.mark.parametrize(
        "text", [FUNCTIONAL, MM_NO_FUNCTIONS, ELEMENTARY], ids=["functional", "mm", "elementary"]
    )
    def test_a_plain_build_emits_no_sens_rhs(self, tmp_path, text):
        """The issue in one assertion, at the emitter — for every rate-law class.

        ``ELEMENTARY`` is the parameter #217 added: under #209 it emitted its
        sensitivity RHS here regardless, because the gate only reached the
        ``functional=`` argument and not the call.
        """
        m = _model(tmp_path, text)
        plain, has_sens = cg.generate_combined_from_model(m, emit_sens_rhs=False)
        assert has_sens is False
        assert "bngsim_codegen_sens_rhs" not in plain

    @pytest.mark.parametrize("text", [FUNCTIONAL, ELEMENTARY], ids=["functional", "elementary"])
    def test_a_sensitivity_build_still_emits_it(self, tmp_path, text):
        m = _model(tmp_path, text)
        m._want_output_sens = True
        src, has_sens = cg.generate_combined_from_model(
            m, emit_output_sens=True, emit_sens_rhs=cg.want_sens_rhs(m)
        )
        assert has_sens is True
        assert "bngsim_codegen_sens_rhs" in src

    def test_the_gate_is_the_request_and_the_hatch_is_separate(self, tmp_path, monkeypatch):
        """``want_sens_rhs`` asks only whether a sensitivity was requested.

        Under #209 it also folded in the GH #67 hatch, because the only thing it
        gated *was* the Functional half. Now that it gates both halves, folding the
        hatch in would make ``BNGSIM_NO_FUNCTIONAL_SENS_RHS=1`` silently switch off
        an Elementary model's sensitivity RHS too — a much bigger hammer than the
        A/B it is documented to be.
        """
        m = _model(tmp_path, FUNCTIONAL)
        assert cg.want_sens_rhs(m) is False, "nobody asked"
        m._want_output_sens = True
        assert cg.want_sens_rhs(m) is True
        monkeypatch.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
        assert cg.want_sens_rhs(m) is True, "the GH #67 hatch is not this question"

    def test_the_hatch_still_only_takes_the_functional_half(self, tmp_path, monkeypatch):
        """With the hatch set, a sensitivity run keeps the Elementary sens RHS and
        loses only the Functional one — the pre-#67 behaviour it exists to restore."""
        monkeypatch.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
        elem = _model(tmp_path, ELEMENTARY, name="e.net")
        elem._want_output_sens = True
        src, has_sens = cg.generate_combined_from_model(elem, emit_sens_rhs=True)
        assert has_sens is True and "bngsim_codegen_sens_rhs" in src

        func = _model(tmp_path, FUNCTIONAL, name="f.net")
        func._want_output_sens = True
        fsrc, fhas = cg.generate_combined_from_model(func, emit_sens_rhs=True)
        assert fhas is False and "bngsim_codegen_sens_rhs" not in fsrc

    def test_the_net_text_emitter_is_gated_too(self, tmp_path):
        """``generate_combined_c``'s own half. The .net text emitter produces the
        Elementary sens RHS without ever consulting the model, so gating
        ``generate_sens_from_model`` alone left it emitting on every plain build —
        the 55.6% #217 measured.
        """
        m = _model(tmp_path, ELEMENTARY)
        net = str(tmp_path / "m.net")
        gated, gated_has = cg.generate_combined_c(net, m, emit_sens_rhs=False)
        ungated, ungated_has = cg.generate_combined_c(net, m, emit_sens_rhs=True)
        assert gated_has is False and ungated_has is True
        assert "bngsim_codegen_sens_rhs" not in gated
        assert "bngsim_codegen_sens_rhs" in ungated
        assert len(gated) < len(ungated)


# ─── the cache keys, both paths ────────────────────────────────────────────


class TestTheCacheKeyCarriesTheFlag:
    def test_the_structural_key_moves_with_the_flag(self, tmp_path):
        """The key follows the flag the caller will actually pass the generator.

        ``compute_model_codegen_hash`` folds in the *resolved* decision rather than
        the raw GH #67 hatch, the rule its own ``chunk_policy`` comment states. It
        is redundancy today — ``emit_output_sens`` determines the flag, so the two
        keys already differ — and it is the cheap kind: the thing it guards is a
        plain-run ``.so`` with no ``bngsim_codegen_sens_rhs`` in it being served to
        a sensitivity run, which is silent.
        """
        m = _model(tmp_path, FUNCTIONAL)
        assert cg.compute_model_codegen_hash(
            m, emit_sens_rhs=True
        ) != cg.compute_model_codegen_hash(m, emit_sens_rhs=False)

    def test_plain_and_sensitivity_were_already_separate_entries(self, tmp_path):
        """The objection #217 was filed holding open, measured.

        Gating the Elementary half was held back because it would make every
        model's plain artifact differ from its sensitivity artifact "where today
        only Functional/MM ones do", roughly doubling entries in an already 2 GB
        cache (issue #205). It does not: #177's ``:sens_term_scale`` has been in
        both keys since before #209, so plain and sensitivity have had separate
        entries — and separate *sources* — for every model, including this
        all-Elementary one. What #217 changes is what is in the plain entry.

        Asserted on ELEMENTARY specifically because that is the class the objection
        was about; a Functional model would separate on the gate itself.
        """
        m = _model(tmp_path, ELEMENTARY)
        m._want_output_sens = False
        plain_key = cg.compute_model_codegen_hash(m, emit_output_sens=False, emit_sens_rhs=True)
        m._want_output_sens = True
        sens_key = cg.compute_model_codegen_hash(m, emit_output_sens=True, emit_sens_rhs=True)
        assert plain_key != sens_key

    def test_nobody_asked_and_the_hatch_do_not_share_a_key(self, tmp_path, monkeypatch):
        """The collision the widening opens, closed.

        Under #209 "nobody asked" and "the GH #67 hatch is set" emitted the same
        source, so one boolean covered both and they shared the
        ``:no_functional_sens`` namespace. For an Elementary model that stopped
        being true the moment the gate covered its half: the hatch leaves the sens
        RHS in, and nobody-asked takes it out. Sharing a key across that is the
        issue #51 inertness trap — a sensitivity run handed a ``.so`` whose
        ``bngsim_codegen_sens_rhs`` is simply absent, no error, just a difference
        quotient.
        """
        m = _model(tmp_path, ELEMENTARY)
        m._want_output_sens = True
        monkeypatch.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
        hatched = cg.compute_model_codegen_hash(m, emit_output_sens=True, emit_sens_rhs=True)
        monkeypatch.delenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", raising=False)
        nobody_asked = cg.compute_model_codegen_hash(m, emit_output_sens=True, emit_sens_rhs=False)
        assert hatched != nobody_asked

    @requires_cc
    def test_the_model_path_compiles_two_distinct_artifacts(self, tmp_path, monkeypatch):
        """End to end for ``prepare_model_codegen``: it must resolve the flag and
        hand the SAME value to the key and to the generator. (The two artifacts
        would be distinct on ``emit_output_sens`` alone; what this pins is that the
        plain one really has no ∂f/∂p and the sensitivity one really does.)"""
        monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cache")
        m = _sbml_functional(tmp_path)
        plain_so = cg.prepare_model_codegen(m)
        m._want_output_sens = True
        sens_so = cg.prepare_model_codegen(m)
        assert plain_so is not None and sens_so is not None
        assert plain_so != sens_so
        assert not _emits_sens_rhs(plain_so)
        assert _emits_sens_rhs(sens_so)

    @requires_cc
    def test_the_net_path_compiles_two_distinct_artifacts(self, tmp_path, monkeypatch):
        """Same, for ``prepare_codegen``. The .net key is built from the file's
        bytes plus cheap flags rather than from the source, so every flag that
        changes the source has to be in it."""
        monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cache")
        cg._PREPARE_CODEGEN_MEMO.clear()
        m = _model(tmp_path, FUNCTIONAL)
        net = str(tmp_path / "m.net")
        plain_so = cg.prepare_codegen(net, m)
        m._want_output_sens = True
        sens_so = cg.prepare_codegen(net, m)
        assert plain_so != sens_so
        assert not _emits_sens_rhs(plain_so)
        assert _emits_sens_rhs(sens_so)

    def test_the_flag_still_shares_the_hatch_namespace_for_a_functional_model(
        self, tmp_path, monkeypatch
    ):
        """Where the #209 sharing survives, it is kept.

        For a **Functional** model the hatch and nobody-asking still emit the same
        source — no sens RHS either way — so they may still share a key rather than
        splitting an already 2 GB cache (issue #205). Only the Elementary case
        needed the split above. Keeping the two apart is what lets the ``.net``
        suffix stay ``:no_functional_sens`` for the hatch and spend a new
        ``:no_sens_rhs`` only on the case that actually differs.
        """
        m = _model(tmp_path, FUNCTIONAL)
        monkeypatch.delenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", raising=False)
        gated, _ = cg.generate_combined_from_model(m, emit_sens_rhs=False)
        monkeypatch.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
        hatched, _ = cg.generate_combined_from_model(m, emit_sens_rhs=True)
        assert gated == hatched


# ─── the second-order effect: the -O tier the RHS is compiled at ───────────


class TestTheOptimizationLevel:
    def test_dead_sens_rhs_can_hold_the_rhs_at_a_lower_opt_tier(self):
        """Why source size was worth gating even though it costs no derivation.

        ``_resolve_opt_flag`` picks its tier from the size of the whole translation
        unit, so a symbol nothing calls drags the RHS that everything calls down
        with it. On the 20 largest ``.net`` models #217 measured five such flips —
        ``fceri_fyn`` compiling its plain RHS at ``-O0`` because of 4.8 MB of dead
        sensitivity source, and four models dropping ``-O1`` where ``-O3`` was
        available.

        Pinned as a property of ``_resolve_opt_flag`` rather than against a corpus
        model, so it holds in a checkout with no ``.net`` corpus: some pair of
        sizes either side of a tier boundary must resolve differently, or removing
        the dead weight buys nothing and the size argument in
        :func:`bngsim._codegen.want_sens_rhs` is wrong.
        """
        tiers = {cg._resolve_opt_flag("cc", n) for n in (10_000, 1_000_000, 3_000_000, 12_000_000)}
        assert len(tiers) > 1, (
            "opt tiers are size-derived, so dead source must be able to move them"
        )
        # fceri_fyn's measured pair: 10,044,927 chars now, 5,208,268 once gated.
        assert cg._resolve_opt_flag("cc", 10_044_927) != cg._resolve_opt_flag("cc", 5_208_268)


# ─── what must NOT silently drop to the difference quotient ────────────────


class TestNoSilentDifferenceQuotient:
    @requires_cc
    def test_the_constructor_path_gets_the_analytic_rhs(self, tmp_path):
        m = _model(tmp_path, FUNCTIONAL)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["kmax"], codegen=True)
        assert _emits_sens_rhs(sim._codegen_so_path or sim._codegen_c_source)

    @requires_cc
    def test_a_plain_artifact_is_not_reused_for_a_sensitivity_run(self, tmp_path):
        """The constructor's reuse block. ``_auto_codegen_for_sensitivity`` no-ops
        the moment an artifact is attached, so inheriting the plain one is silent:
        no exception, no warning, just a difference quotient. Live before #209 for
        ``bngsim_codegen_output_sens``; #209 would have added ``∂f/∂p`` to it."""
        m = _model(tmp_path, FUNCTIONAL)
        bngsim.Simulator(m, method="ode", codegen=True)
        assert m._codegen_so_path or m._codegen_c_source, "the plain artifact to shadow with"
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["kmax"])
        assert _emits_sens_rhs(sim._codegen_so_path or sim._codegen_c_source)

    @requires_cc
    def test_a_sensitivity_artifact_is_still_reused_by_a_plain_run(self, tmp_path):
        """The converse stays allowed — a sensitivity artifact is a superset, and
        refusing it would cost a rebuild for nothing."""
        m = _model(tmp_path, FUNCTIONAL)
        sens = bngsim.Simulator(m, method="ode", sensitivity_params=["kmax"], codegen=True)
        plain = bngsim.Simulator(m, method="ode")
        assert plain._codegen_so_path == sens._codegen_so_path
        assert plain._codegen_c_source == sens._codegen_c_source

    @requires_cc
    @pytest.mark.parametrize(
        "text,param",
        [(FUNCTIONAL, "kmax"), (MM_NO_FUNCTIONS, "kcat"), (ELEMENTARY, "k1")],
        ids=["functional", "michaelis-menten-no-functions", "elementary"],
    )
    def test_a_method_argument_request_regenerates(self, tmp_path, text, param):
        """``steady_state(sensitivity_params=...)`` takes its request as a METHOD
        argument, so the constructor never set ``_want_output_sens`` and the
        artifact it built carries no ∂f/∂p.

        ``MM_NO_FUNCTIONS`` is the case that made this more than bookkeeping: the
        regeneration used to be gated on ``n_functions > 0``, and a
        Michaelis-Menten model has none, so it would have kept the plain artifact
        and reported ``sens_dfdp_source == "finite-difference"`` with no error.
        """
        m = _model(tmp_path, text, name=f"{param}.net")
        sim = bngsim.Simulator(m, method="ode", codegen=True)
        ss = sim.steady_state(sensitivity_params=[param], max_time=1e5)
        assert ss.sens_dfdp_source == "codegen"

    @requires_cc
    def test_compute_all_sensitivities_regenerates(self, tmp_path):
        m = _model(tmp_path, FUNCTIONAL)
        sim = bngsim.Simulator(m, method="ode", codegen=True)
        assert not _emits_sens_rhs(sim._codegen_so_path or sim._codegen_c_source)
        sim.compute_all_sensitivities(t_span=(0.0, 1.0), n_points=3)
        assert _emits_sens_rhs(sim._codegen_so_path or sim._codegen_c_source)
