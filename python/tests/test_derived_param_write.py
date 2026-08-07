"""Issue #188 — ``set_param`` on a *derived* (expression-backed) parameter.

A derived parameter's value comes from its defining expression, not from
storage: ``d`` in ``d = d__FREE``, or one of the ``_rateLaw{N}`` symbols BNG2.pl
and the SBML loader synthesize for a compound rate law. Writing one directly is
an **override** — BNG's ``setParameter`` semantics, and deliberate: ``d`` stops
tracking ``d__FREE`` for the remainder of the action sequence.
:mod:`test_validation` and :mod:`test_new_features` own that contract and it is
unchanged here.

What #188 reported is that the override was *latched* — flipped by the act of
writing rather than keyed on the value — which made it both too eager and
irreversible:

* **Too eager.** A write of the value the parameter already held counted as an
  override. So ``set_params(dict(zip(param_names, vec)))`` — the whole
  parameter-vector round trip a fitting harness performs every iteration, and
  the one :meth:`Result.gradient`'s own docstring recommends — overrode every
  derived parameter in the model.
* **Irreversible.** Writing the original value back did not restore anything,
  and neither did :meth:`reset`, :meth:`clone`, or writing the primary
  underneath. The evaluator was discarded on the first write.

And an override is not bookkeeping. A derived parameter pinned to a literal no
longer carries the chain rule from the primaries underneath it, so those
primaries lose that reaction's term from ``∂f/∂p`` and the generated C changes to
match. That is *correct* for a deliberate override — but under the latch it also
happened to callers who only round-tripped a vector, and on BIOMD0000000701 it
took ``d(x)/d(alpha)`` from 1.0e-3 to exactly zero for a parameter never written.
``test_identity_vector_write_does_not_zero_a_primary_sensitivity`` is that
measurement, and it is the sharpest statement of why this is a correctness bug
rather than a caching one.

The rule now: **a parameter with a defining expression is expression-backed
exactly while it holds the value that expression produces.** Exact equality, and
exact by construction rather than by luck — the re-evaluation loop in
``NetworkModel::set_param`` is the only writer of an attached derived value, so
an attached parameter always holds precisely ``evaluate(expression)``. That was
checked bit-for-bit across the 9,524 derived parameters in the 279 rr_parity
corpus models that have any, which is what licenses ``==`` here instead of a
tolerance.

The invariant these tests defend, in one line: *writing a derived parameter and
writing it back restores the generated source*, with the primary-parameter
control on the same model as the contrast — a primary write never reaches the
source at all.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

_REPRO = Path(__file__).resolve().parents[2] / "benchmarks" / "sbml_events" / "BIOMD0000000701.xml"
_MODELS_DIR = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"
# The corpus is gitignored, so the directory itself exists in a fresh worktree
# (and in CI) while holding nothing. Gating on `is_dir()` is how a corpus test
# turns into a silent no-op in exactly the checkouts that are not the author's —
# see #192, which shipped a 77-model regression behind that mistake. Gate on a
# model actually being there, and make the sweep assert it reached one.
_CORPUS_MODELS = sorted(_MODELS_DIR.glob("*/*.xml")) if _MODELS_DIR.is_dir() else []


def _source_hash(model: bngsim.Model) -> str:
    """SHA-256 of the generated C, sensitivity included.

    The sensitivity half is where a derived parameter's chain rule lives, so a
    plain RHS hash would not see an override at all.
    """
    from bngsim import _codegen as cg

    model._want_output_sens = True
    src, *_ = cg.generate_combined_from_model(model, emit_output_sens=True)
    return hashlib.sha256(src.encode()).hexdigest()


def _derived_names(model: bngsim.Model) -> list[str]:
    return [n for n, f in zip(model.param_names, model.param_is_expression, strict=True) if f]


def _attached(model: bngsim.Model, name: str) -> bool:
    return model.param_is_expression[list(model.param_names).index(name)]


# ── The core contract, on a self-contained model ────────────────────────────
#
# `k = 2 * k_base` is the whole of `d = d__FREE` with a coefficient, and needs
# no fixture file, so a failure here is a failure of the rule itself rather than
# of anything a loader did.


def _two_param_model() -> bngsim.Model:
    b = ModelBuilder()
    b.add_parameter("k_base", 0.5)
    b.add_parameter("k", 0.0, expression="2 * k_base", is_expression=True)
    s_idx = b.add_species("S", 100.0)
    b.add_reaction([s_idx], [], "elementary", "k")
    return bngsim.Model(b.build())


def test_identity_write_to_a_derived_parameter_is_not_an_override():
    """Writing the value it already holds leaves a derived parameter derived.

    This is the half that made the whole-vector round trip destructive: under
    the latch, ``set_param("k", model.get_param("k"))`` detached ``k``, and
    nothing about the call said so.
    """
    m = _two_param_model()
    assert _attached(m, "k") and m.get_param("k") == pytest.approx(1.0)

    m.set_param("k", m.get_param("k"))

    assert _attached(m, "k"), "an identity write must not override the expression"
    # Still tracking: the proof that it is attached is that a primary moves it.
    m.set_param("k_base", 3.0)
    assert m.get_param("k") == pytest.approx(6.0)


def test_a_value_changing_write_still_overrides_bng_style():
    """The BNG ``setParameter`` contract is unchanged: the literal wins.

    Kept next to the reversibility test on purpose — the pair is the whole
    design, and a future simplification that makes the round trip work by
    ignoring overrides entirely has to break this one to do it.
    """
    m = _two_param_model()
    m.set_param("k", 27.0)

    assert not _attached(m, "k")
    assert m.get_param("k") == pytest.approx(27.0)

    # The primary underneath no longer reaches it.
    m.set_param("k_base", 100.0)
    assert m.get_param("k") == pytest.approx(27.0)
    # ...and the expression is still on record, which is how a caller tells an
    # overridden derived parameter from a genuine primary.
    assert m._core.param_expressions[list(m.param_names).index("k")] == "2 * k_base"


def test_an_override_is_lifted_by_writing_back_the_expressions_value():
    """The way back. Under the latch there was none — not reset(), not clone()."""
    m = _two_param_model()
    m.set_param("k", 27.0)
    assert not _attached(m, "k")

    m.set_param("k", 1.0)  # == 2 * k_base

    assert _attached(m, "k"), "writing the expression's own value must re-attach"
    m.set_param("k_base", 4.0)
    assert m.get_param("k") == pytest.approx(8.0), "re-attached, so the primary moves it"


def test_reset_and_clone_preserve_an_override_and_keep_it_reversible():
    """An override survives reset() and clone() — and stays liftable in both.

    ``clone()`` re-compiles parameter expressions into the copy's own evaluator,
    and it used to skip any parameter that was not currently attached. That made
    a clone the one copy of the model that could never be put back, silently.
    """
    m = _two_param_model()
    m.set_param("k", 27.0)
    m.reset()
    assert not _attached(m, "k") and m.get_param("k") == pytest.approx(27.0)

    c = m.clone()
    assert not _attached(c, "k") and c.get_param("k") == pytest.approx(27.0)

    c.set_param("k", 1.0)
    assert _attached(c, "k"), "the clone must be able to lift the override too"
    c.set_param("k_base", 4.0)
    assert c.get_param("k") == pytest.approx(8.0)
    # The original is untouched by the clone's re-attach.
    assert not _attached(m, "k")


def test_force_override_pins_a_derived_parameter_against_an_identity_write():
    """``force_override=True`` is the "treat this as an independent input" escape hatch.

    The rule that makes an ordinary write round-trip also takes something away:
    a caller whose whole contract is "every parameter is its own axis" used to
    get that for free from the unconditional detach, and an identity write no
    longer supplies it. :func:`bngsim.jax.differentiable_solve` with
    ``flat=True`` is that caller — without the pin, writing ``_rateLaw1`` its
    own nominal value leaves it tracking ``kon``, and ``jax.grad`` returns a
    non-zero gradient for a coordinate it was told is independent.

    The pin is deliberately permanent for the model object: that is the legacy
    behaviour exactly, and a pin a later identity write could lift would not be
    one.
    """
    m = _two_param_model()
    m.set_param("k", m.get_param("k"), force_override=True)

    assert not _attached(m, "k"), "force_override must pin even on an identity write"
    m.set_param("k_base", 3.0)
    assert m.get_param("k") == pytest.approx(1.0), "pinned, so the primary must not move it"

    # ...and unlike an ordinary override, writing the expression's value back
    # does not lift it.
    m.set_param("k", 6.0)  # == 2 * k_base now
    assert not _attached(m, "k")
    assert m.get_param("k") == pytest.approx(6.0)

    # It is also inert on a parameter that has no expression to override.
    m.set_param("k_base", 9.0, force_override=True)
    assert not _attached(m, "k_base")
    assert m.get_param("k_base") == pytest.approx(9.0)


# ── The reported reproducer, inverted ───────────────────────────────────────


@pytest.mark.skipif(not _REPRO.exists(), reason=f"benchmark model not present: {_REPRO}")
def test_derived_write_round_trips_the_generated_source():
    """#188's reproducer, as an assertion: perturb, restore, hash returns.

    Three derived parameters, each perturbed off its value and written back. The
    filed run reported ``source_changed=True restores=False`` for all three.
    """
    m = bngsim.Model.load(str(_REPRO))
    derived = _derived_names(m)
    assert len(derived) >= 3, f"expected the reproducer's derived parameters, got {derived}"

    base = _source_hash(m)
    for name in derived[:3]:
        original = m.get_param(name)
        m.set_param(name, original * 3.7 + 1.3)
        assert not _attached(m, name), f"{name}: a value-changing write must override"

        m.set_param(name, original)

        assert _attached(m, name), f"{name}: writing the original back must re-attach"
        assert _source_hash(m) == base, f"{name}: the generated source did not round-trip"


@pytest.mark.skipif(not _REPRO.exists(), reason=f"benchmark model not present: {_REPRO}")
def test_primary_write_never_reaches_the_generated_source():
    """The control that makes the contrast sharp.

    A primary parameter is a ``p[]`` read at run time, so no value of it changes
    a byte of the generated C. That has always held — it is here so that a
    regression which moved the source for *every* write could not pass the test
    above by making both sides equally wrong.
    """
    m = bngsim.Model.load(str(_REPRO))
    name = m.primary_param_names[0]
    base = _source_hash(m)

    original = m.get_param(name)
    m.set_param(name, original * 3.7 + 1.3)
    assert _source_hash(m) == base, "a primary write must not move the source at all"

    m.set_param(name, original)
    assert _source_hash(m) == base


@pytest.mark.skipif(not _REPRO.exists(), reason=f"benchmark model not present: {_REPRO}")
def test_identity_vector_write_leaves_every_derived_parameter_attached():
    """``set_params(dict(zip(param_names, vec)))`` is a no-op, including structurally.

    The idiom is straight out of :meth:`Result.gradient`'s docstring — the
    scipy ``minimize`` objective it recommends runs it once per iteration.
    """
    m = bngsim.Model.load(str(_REPRO))
    before_attached = list(m.param_is_expression)
    before_primaries = list(m.primary_param_names)
    base = _source_hash(m)
    assert any(before_attached), "this model must have derived parameters to be a test"

    m.set_params(dict(zip(m.param_names, [m.get_param(n) for n in m.param_names], strict=True)))

    assert list(m.param_is_expression) == before_attached
    assert list(m.primary_param_names) == before_primaries
    assert _source_hash(m) == base


@pytest.mark.skipif(not _REPRO.exists(), reason=f"benchmark model not present: {_REPRO}")
def test_identity_vector_write_does_not_zero_a_primary_sensitivity():
    """The consequence that makes this a correctness bug, not a caching one.

    ``alpha`` reaches the trajectory only through ``_rateLaw_R1*_fwd =
    alpha * kon*``. Override those and ``alpha`` genuinely stops mattering — so
    under the latch a caller who round-tripped the parameter vector got
    ``d(x)/d(alpha) == 0`` exactly, for a parameter they never wrote, with no
    warning and an otherwise identical model.
    """
    m = bngsim.Model.load(str(_REPRO))
    assert "alpha" in m.param_names

    ref = np.asarray(
        bngsim.Simulator(m)
        .compute_all_sensitivities((0.0, 100.0), 21, params=["alpha"])
        .sensitivities
    )
    assert np.abs(ref).max() > 0.0, "alpha must have a non-zero column for this to bite"

    m.set_params(dict(zip(m.param_names, [m.get_param(n) for n in m.param_names], strict=True)))
    after = np.asarray(
        bngsim.Simulator(m)
        .compute_all_sensitivities((0.0, 100.0), 21, params=["alpha"])
        .sensitivities
    )

    np.testing.assert_allclose(after, ref, rtol=1e-8, atol=0.0)


# ── Corpus sweep: the invariant, everywhere it applies ───────────────────────


@pytest.mark.skipif(not _CORPUS_MODELS, reason=f"rr_parity corpus not present: {_MODELS_DIR}")
def test_corpus_derived_writes_round_trip_the_source():
    """Every derived parameter in the corpus: write, restore, source hash returns.

    Cheap — no integrator, just codegen — and it is the invariant in its most
    general form. Bounded to the first few models carrying derived parameters
    and the first few parameters in each: codegen is the cost here, and the
    mechanism is per-parameter, so breadth across models buys more than depth
    within one.

    #188 recorded this sweep shape as a measurement trap in its own right: under
    the latch, the first non-restoring derived write made *every subsequent*
    probe read as changed — 114 of 135 spurious "primary" hits on
    BIOMD0000000571 — so a sweep that walks parameters without asserting the
    restore reports noise. Asserting it per parameter is what keeps that from
    recurring silently.
    """
    checked_models = 0
    checked_params = 0
    failures: list[str] = []

    for xml in _CORPUS_MODELS:
        if checked_models >= 12:
            break
        try:
            m = bngsim.Model.load(str(xml))
            derived = _derived_names(m)
            if not derived:
                continue
            base = _source_hash(m)
        except Exception:  # a corpus model this build cannot load is not this test's subject
            continue

        checked_models += 1
        for name in derived[:4]:
            original = m.get_param(name)
            try:
                m.set_param(name, original * 3.7 + 1.3)
                m.set_param(name, original)
                restored = _source_hash(m)
            except Exception as exc:  # noqa: BLE001 - reported, not raised, so the sweep completes
                failures.append(f"{xml.parent.name}:{name}: raised {exc!r}")
                continue
            checked_params += 1
            if restored != base:
                failures.append(f"{xml.parent.name}:{name}: source did not round-trip")
            if not _attached(m, name):
                failures.append(f"{xml.parent.name}:{name}: left overridden after restore")

    assert checked_models > 0, "no corpus model with a derived parameter was reached"
    assert checked_params > 0
    assert not failures, f"{len(failures)} derived write(s) did not round-trip: {failures[:10]}"
