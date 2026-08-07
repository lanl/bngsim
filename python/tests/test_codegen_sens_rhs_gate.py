"""Issue #209 — the analytic ∂f/∂p is derived only when a sensitivity is asked for.

``generate_combined_from_model`` called ``generate_sens_from_model`` unconditionally,
so ``Simulator(model, method="ode")`` with no ``sensitivity_params`` derived the
Functional sensitivity RHS with sympy, emitted it, and compiled it into the cached
``.so`` — for a run that never calls ``CVodeSensInit1``. On ``BIOMD0000000496``
(295 species, 333 functional reactions, cold cache) that was 39.5 s of construction
and a 26.7 MB ``.so``, against 21.5 s and 1.8 MB once the derivation is skipped.
The GH #198 output-sensitivity evaluator three lines below in the same function was
already gated on ``_want_output_sens`` for exactly this reason; ``∂f/∂p`` is the
more expensive of the two and was the one left ungated.

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
# FUNCTIONAL is the class the gate is FOR (its ∂f/∂p is the sympy derivation);
# MM_NO_FUNCTIONS is the class the gate's plumbing is most likely to drop, because
# a Michaelis-Menten model needs no functions at all and the regeneration trigger
# used to key on ``n_functions > 0``; ELEMENTARY is the class that must not move —
# its sensitivity RHS is plain text emission with no sympy in it, so #209 leaves
# it unconditional and its source (and cached .so) byte-identical.

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
    def test_a_plain_build_does_not_derive_the_functional_sens_rhs(self, tmp_path):
        """The issue in one assertion, at the emitter."""
        m = _model(tmp_path, FUNCTIONAL)
        plain, has_sens = cg.generate_combined_from_model(m, emit_functional_sens=False)
        assert has_sens is False
        assert "bngsim_codegen_sens_rhs" not in plain

    def test_a_sensitivity_build_still_derives_it(self, tmp_path):
        m = _model(tmp_path, FUNCTIONAL)
        m._want_output_sens = True
        src, has_sens = cg.generate_combined_from_model(
            m, emit_output_sens=True, emit_functional_sens=cg.want_functional_sens_rhs(m)
        )
        assert has_sens is True
        assert "bngsim_codegen_sens_rhs" in src

    def test_the_resolved_flag_is_the_hatch_and_the_request(self, tmp_path, monkeypatch):
        """``want_functional_sens_rhs`` is the GH #67 process hatch AND the
        per-run question. Either one off is off."""
        m = _model(tmp_path, FUNCTIONAL)
        assert cg.want_functional_sens_rhs(m) is False, "nobody asked"
        m._want_output_sens = True
        assert cg.want_functional_sens_rhs(m) is True
        monkeypatch.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
        assert cg.want_functional_sens_rhs(m) is False, "the GH #67 hatch still wins"

    def test_an_elementary_model_is_untouched(self, tmp_path):
        """#209 gates only the Functional/MM half. An Elementary sensitivity RHS
        is text emission with no sympy in it, so its source — and every ``.so``
        cached for it — must not move."""
        m = _model(tmp_path, ELEMENTARY)
        gated, gated_has = cg.generate_combined_from_model(m, emit_functional_sens=False)
        ungated, ungated_has = cg.generate_combined_from_model(m, emit_functional_sens=True)
        assert gated_has is True and ungated_has is True
        assert gated == ungated


# ─── the cache keys, both paths ────────────────────────────────────────────


class TestTheCacheKeyCarriesTheFlag:
    def test_the_structural_key_moves_with_the_flag(self, tmp_path):
        """A key that ignored it would hand the plain-run ``.so`` — which has no
        ``bngsim_codegen_sens_rhs`` in it at all — to a sensitivity run."""
        m = _model(tmp_path, FUNCTIONAL)
        assert cg.compute_model_codegen_hash(
            m, emit_functional_sens=True
        ) != cg.compute_model_codegen_hash(m, emit_functional_sens=False)

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

    def test_the_flag_shares_the_hatch_namespace(self, tmp_path, monkeypatch):
        """A build with the GH #67 hatch SET and a build with nobody asking emit
        the same source — no sens RHS — so they may share one key rather than
        splitting the cache. Pinned because the alternative is a silent doubling
        of an already 2 GB cache (issue #205)."""
        m = _model(tmp_path, FUNCTIONAL)
        monkeypatch.delenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", raising=False)
        gated, _ = cg.generate_combined_from_model(m, emit_functional_sens=False)
        monkeypatch.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
        hatched, _ = cg.generate_combined_from_model(m, emit_functional_sens=True)
        assert gated == hatched


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
        [(FUNCTIONAL, "kmax"), (MM_NO_FUNCTIONS, "kcat")],
        ids=["functional", "michaelis-menten-no-functions"],
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
