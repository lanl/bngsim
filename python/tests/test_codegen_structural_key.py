"""Issue #174 — the model-side ``.so`` cache key is structural, not a source hash.

``prepare_model_codegen`` used to derive its cache key by generating the C source
and SHA-256'ing it. That made a warm cache skip only the ``cc`` compile: every
``Simulator`` construction still paid the RHS + ∂f/∂p + Jacobian derivation that
is 97% of construction on ``Smith_BMCSystBiol2013``, and none of it depends on
the parameter values a fit is moving. :func:`bngsim._codegen.compute_model_codegen_hash`
now derives the key from cheap C++ reads instead, and the cache hit returns
without generating anything.

Two properties have to hold, and they are not equally serious:

* **No collisions.** Two model states with the same key but different generated
  source would load each other's ``.so`` — a silently wrong RHS, not a slow
  build. The bar is zero, and the sharp case is not hypothetical:
  :func:`test_a_parameter_value_that_flips_the_switch_gate_moves_the_key` builds
  a model where writing one *value* changes the emitted C by 4 KB.
* **Few spurious differences.** A key that moves when the source does not costs
  one cache miss — today's behaviour, never worse. Measured, not gated.

The corpus sweep behind those claims is in the PR; what is pinned here is the
mechanism, the injectivity of the serializer, and the one invariant a future
refactor is most likely to break — that computing the key generates no source.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import bngsim
import pytest
from bngsim import _codegen as cg
from bngsim._bngsim_core import ModelBuilder
from bngsim._switch_sensitivity import switch_gate_cache_digest

_MODELS_DIR = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"
# Gate on a model FILE, never on `is_dir()`: `models/` self-ignores through its
# own tracked .gitignore, so the directory exists — empty — in every worktree and
# in CI. #192 shipped a 77-model regression behind exactly that mistake.
_CORPUS_MODELS = sorted(_MODELS_DIR.glob("*/*.xml")) if _MODELS_DIR.is_dir() else []


def _src_hash(model, *, emit_output_sens: bool = False) -> str:
    src, _ = cg.generate_combined_from_model(model, emit_output_sens=emit_output_sens)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _decay_model() -> bngsim.Model:
    """``k = 2 * k_base`` over one decaying species — a primary, a derived
    parameter that a rate law actually reads, and nothing else."""
    b = ModelBuilder()
    b.add_parameter("k_base", 0.5)
    b.add_parameter("k", 0.0, expression="2 * k_base", is_expression=True)
    s = b.add_species("S", 100.0)
    b.add_reaction([s], [], "elementary", "k")
    return bngsim.Model(b.build())


def _clock_gate_model(clock_rate: float) -> bngsim.Model:
    """A counter species whose slope IS a parameter, gating an equality condition.

    ``dT/dt = k_clock``, so ``T`` is a unit-rate clock exactly when
    ``k_clock == 1``. The rate law's condition is an *equality*, which
    ``state_switch_residual`` refuses to root on — so the equality is
    compensated only through the clock path, and only while ``T`` is a clock.
    Writing ``k_clock`` therefore decides whether the analytic sensitivity RHS is
    emitted at all: a parameter VALUE moving the generated source.
    """
    b = ModelBuilder()
    b.add_parameter("k_clock", clock_rate)
    b.add_parameter("kd", 0.3)
    t = b.add_species("T", 0.0)
    s = b.add_species("S", 10.0)
    b.add_reaction([], [t], "elementary", "k_clock")
    b.add_observable("Tobs", [(t, 1.0)])
    b.add_function("gate", "if(Tobs == 5, kd, 0)")
    b.add_reaction([s], [], "functional", "gate")
    return bngsim.Model(b.build())


# ── The serializer: the most likely source of a silent collision ─────────────


@pytest.mark.parametrize(
    "left,right,why",
    [
        ({"a": "b:c"}, {"a:b": "c"}, "a delimiter inside a string must not read as structure"),
        ([1, [2]], [[1], 2], "nesting must not flatten"),
        ([[1, 2], [3]], [[1], [2, 3]], "a list boundary must survive"),
        (0.0, -0.0, "IEEE-754 signed zero is two distinct values"),
        (1, 1.0, "an int and a float are different C literals"),
        (True, 1, "bool is an int subclass and must still be distinguishable"),
        ({"a": 1}, {"a": 1, "b": None}, "an absent key is not a null one"),
        (["ab"], ["a", "b"], "concatenation must not alias a split"),
    ],
)
def test_canonical_serialization_separates(left, right, why):
    assert _canon(left) != _canon(right), why


def test_canonical_serialization_ignores_dict_insertion_order():
    assert _canon({"a": 1, "b": 2}) == _canon({"b": 2, "a": 1})


def test_canonical_serialization_refuses_an_unknown_type():
    """Raising beats hashing ``repr()``: an object whose repr carries its id()
    would give the same model two keys per process, and one that does not would
    give two models one key."""
    with pytest.raises(TypeError, match="cannot serialize"):
        _canon(object())


def _canon(obj) -> str:
    h = hashlib.sha256()
    cg._canon_update(h, obj)
    return h.hexdigest()


# ── The point of the issue: the key costs no source generation ───────────────


def test_computing_the_key_never_generates_source():
    """The whole issue in one assertion.

    A key derived from the generated source cannot skip generating it, so this
    is the invariant a future refactor would have to break to reintroduce #174.
    Enforced by making generation fatal rather than by timing it.
    """
    m = _decay_model()

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("computing the cache key generated C source")

    saved = cg.generate_combined_from_model
    cg.generate_combined_from_model = explode
    try:
        assert cg.compute_model_codegen_hash(m)
    finally:
        cg.generate_combined_from_model = saved


def test_a_warm_cache_returns_the_so_without_generating_source(tmp_path, monkeypatch):
    """End to end: the second ``prepare_model_codegen`` on an unchanged model
    resolves the cached ``.so`` with source generation disabled."""
    # CACHE_DIR is resolved from the environment at import, so the env var is
    # too late here — patch the module global the cache actually reads.
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
    m = _decay_model()
    first = cg.prepare_model_codegen(m)
    if first is None:  # pragma: no cover - a toolchain without a working cc
        pytest.skip("codegen compile unavailable in this environment")

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("a warm cache regenerated the C source")

    monkeypatch.setattr(cg, "generate_combined_from_model", explode)
    assert cg.prepare_model_codegen(_decay_model()) == first


# ── What the key must and must not follow ────────────────────────────────────


def test_a_primary_parameter_value_moves_neither_the_source_nor_the_key():
    """The property the fit relies on. Both halves asserted: a key that ignored a
    value the source *did* read would be a collision, so the source is checked
    too rather than assumed."""
    m = _decay_model()
    key, src = cg.compute_model_codegen_hash(m), _src_hash(m)

    m.set_param("k_base", 41.0)

    assert _src_hash(m) == src, "a primary's value is read from p[] at run time"
    assert cg.compute_model_codegen_hash(m) == key


def test_a_species_initial_condition_moves_neither_the_source_nor_the_key():
    m = _decay_model()
    key, src = cg.compute_model_codegen_hash(m), _src_hash(m)

    m.set_concentration("S", 7.0)

    assert _src_hash(m) == src
    assert cg.compute_model_codegen_hash(m) == key


def test_overriding_a_derived_parameter_moves_the_key_and_writing_it_back_restores_it():
    """The #188 round trip, at the key. An override detaches ``k`` from
    ``2 * k_base``, which drops the chain rule from ∂f/∂p and really does change
    the emitted C — so the key must move with it, and must come back when the
    override is lifted."""
    m = _decay_model()
    key, src = cg.compute_model_codegen_hash(m), _src_hash(m)

    m.set_param("k", 27.0)
    assert _src_hash(m) != src
    assert cg.compute_model_codegen_hash(m) != key

    m.set_param("k", 2 * m.get_param("k_base"))
    assert _src_hash(m) == src
    assert cg.compute_model_codegen_hash(m) == key


def test_output_sensitivity_emission_moves_the_key():
    """``emit_output_sens`` appends a whole callback, so a plain-run ``.so`` must
    never satisfy a sensitivity run (the issue #51 inertness trap)."""
    m = _decay_model()
    assert cg.compute_model_codegen_hash(
        m, emit_output_sens=True
    ) != cg.compute_model_codegen_hash(m, emit_output_sens=False)


# ── The one path by which a parameter VALUE reaches the emitted source ───────


def test_the_switch_gate_digest_is_empty_for_a_condition_free_model():
    """The cost claim: the condition-free majority pays only the text pre-scan."""
    assert switch_gate_cache_digest(_decay_model()._core) == ()


def test_a_parameter_value_that_flips_the_switch_gate_moves_the_key():
    """A parameter value CAN change the generated source, through the issue #68 gate.

    ``_unit_rate_clock_species`` probes the RHS, so whether ``T`` is a clock
    depends on ``k_clock``'s value; and only while it is a clock is the equality
    condition compensated, so only then is the analytic sensitivity RHS emitted.
    One ``set_param`` moves the emitted C by several kilobytes.

    This is the case that makes ``switch_gate_cache_digest`` load-bearing rather
    than defensive — a key built from structure alone would serve the sensitivity
    ``.so`` to the model that must not have one.
    """
    m = _clock_gate_model(1.0)
    key, src = cg.compute_model_codegen_hash(m), _src_hash(m)
    assert switch_gate_cache_digest(m._core) != ()

    m.set_param("k_clock", 2.0)

    assert _src_hash(m) != src, "the gate no longer admits the condition"
    assert cg.compute_model_codegen_hash(m) != key


def test_the_key_would_collide_without_the_switch_gate_digest():
    """The negative control for the test above: with the verdict stubbed out, the
    two states are structurally identical and the key cannot tell them apart.

    Kept so the previous test cannot pass for the wrong reason — if some other
    part of the key happened to separate these models, dropping the digest would
    be free and nobody would notice it had stopped mattering.
    """
    import bngsim._switch_sensitivity as sw

    a, b = _clock_gate_model(1.0), _clock_gate_model(2.0)
    assert _src_hash(a) != _src_hash(b)

    saved = sw.switch_gate_cache_digest
    sw.switch_gate_cache_digest = lambda core, ctx=None: ()
    try:
        assert cg.compute_model_codegen_hash(a) == cg.compute_model_codegen_hash(b)
    finally:
        sw.switch_gate_cache_digest = saved


# ── Two defects a structural key promotes, fixed with it ─────────────────────


def _output_sens_model() -> bngsim.Model:
    """A user function reading a derived parameter — the shape whose ∂func/∂θ
    carries the #15 chain rule that an override removes."""
    b = ModelBuilder()
    b.add_parameter("kb", 0.5)
    b.add_parameter("k", 0.0, expression="2 * kb", is_expression=True)
    s = b.add_species("S", 100.0)
    b.add_observable("Sobs", [(s, 1.0)])
    b.add_function("flux", "k * Sobs")
    b.add_reaction([s], [], "functional", "flux")
    m = bngsim.Model(b.build())
    m._want_output_sens = True
    return m


def test_a_reused_model_emits_the_same_source_as_a_fresh_one_after_a_derived_override():
    """``_analyze_output_sens``'s memo has to follow the attachment vector.

    Its key was four counters and the budget tag, on the stated grounds that
    ``set_param`` only writes values — which #188 falsified for a derived
    parameter. Overriding one drops the chain rule from ``∂func/∂θ`` without
    moving any counter, so a model that had already been emitted kept emitting
    the pre-write partials.

    Content addressing hid this: the ``.so`` was keyed on the stale source it was
    compiled from, so it at least matched itself, and a fresh process derived the
    right thing. Under a structural key the stale source would be cached under the
    *post*-write key and served to everyone after. That is why this is fixed here
    and not filed for later.
    """
    reused = _output_sens_model()
    pristine = _src_hash(reused, emit_output_sens=True)  # populates the memo
    reused.set_param("k", 27.0)

    fresh = _output_sens_model()
    fresh.set_param("k", 27.0)

    assert _src_hash(reused, emit_output_sens=True) != pristine, "the override must reach ∂func/∂θ"
    assert _src_hash(reused, emit_output_sens=True) == _src_hash(fresh, emit_output_sens=True)

    reused.set_param("k", 2 * reused.get_param("kb"))
    assert _src_hash(reused, emit_output_sens=True) == pristine


def test_the_net_cache_key_separates_the_chunking_hatch(monkeypatch, tmp_path):
    """The sibling key site had the same hole, for the fourth hatch (issue #174).

    ``prepare_codegen`` keys on the ``.net``'s bytes plus a suffix per
    process-scoped knob — ``BNGSIM_NO_CODEGEN_JAC``, the GH #67 hatch, the GH #90
    budget — but not on ``BNGSIM_CODEGEN_CHUNK``, which changes what
    ``generate_rhs_c`` emits. So a chunked run was served the unchunked ``.so``:
    a wrong artifact, and an A/B of the feature that measures one binary twice.
    """
    # A TRACKED .net, so this runs in CI too — the gitignored ode_fullnet suite
    # would make it a silent no-op everywhere but the author's checkout.
    nets = sorted(
        (Path(__file__).resolve().parents[2] / "benchmarks/models/net/ode").glob("*.net")
    )
    net = next((p for p in nets if len(cg._parse_net_file(str(p))["reactions"]) >= 30), None)
    assert net is not None, "no tracked .net is large enough to chunk"

    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cg, "_PREPARE_CODEGEN_MEMO", {})

    plain_src = cg.generate_rhs_c(str(net))
    plain_so = cg.prepare_codegen(str(net))

    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "on")
    assert cg.generate_rhs_c(str(net)) != plain_src, "the hatch did not change the emitted source"

    # compute_model_hash is the .net file's bytes and stays put by design; it is
    # prepare_codegen's suffix that has to separate the two artifacts.
    assert cg.prepare_codegen(str(net)) != plain_so


# ── The biconditional, on real models ────────────────────────────────────────


@pytest.mark.skipif(not _CORPUS_MODELS, reason="rr_parity corpus not present")
@pytest.mark.parametrize("path", _CORPUS_MODELS[::97], ids=lambda p: p.parent.name)
def test_corpus_key_matches_source_under_perturbation(path):
    """``key(A) == key(B)`` iff ``source(A) == source(B)``, over a corpus sample.

    Every probe reloads the model: #188 records a sweep whose phases shared a
    mutated model reporting a confident zero on a broken binary. A same-key /
    different-source pair fails hard; the other direction only costs a cache
    miss, so it is not asserted here (the PR measures it over the whole corpus).
    """
    base = bngsim.Model.load(str(path))
    key0, src0 = cg.compute_model_codegen_hash(base), _src_hash(base)
    names = list(base.param_names)
    is_expr = list(base._core.param_is_expression)
    del base

    probes = [n for n, e in zip(names, is_expr, strict=True) if not e][:4]
    probes += [n for n, e in zip(names, is_expr, strict=True) if e][:2]
    assert probes, f"{path.parent.name} exposes no parameter to perturb"

    for name in probes:
        m = bngsim.Model.load(str(path))
        try:
            m.set_param(name, (m.get_param(name) or 1.0) * 1.7)
        except (ValueError, RuntimeError):
            continue  # a refused write (a compartment size) is not a probe
        if cg.compute_model_codegen_hash(m) == key0:
            assert _src_hash(m) == src0, (
                f"COLLISION on {path.parent.name}: writing {name!r} changed the generated "
                "source but not the cache key — the wrong .so would be loaded"
            )
