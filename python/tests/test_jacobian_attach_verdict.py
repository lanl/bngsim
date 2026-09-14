"""Issue #534: the C++ attach gate's verdict reaches Python as text.

``NetworkModel::set_functional_jacobian`` returns a bool, and when it declined the
reason existed only as a stderr line under ``BNGSIM_JAC_DEBUG=1`` — which probe,
which entry, both values — so ``Model.analytical_jacobian_status`` read the same
generic sentence for every refusal. On main at 5f55012 that was 17 corpus models, in
two classes invisible from the status: eleven whose analytical entry is non-finite
at the seed while the RHS is finite (``∂(k·sqrt(A))/∂A`` at ``A = 0``), and six with
a mismatch, five of them with their seed on a switching surface of their own
``if(c > 0)`` / ``if(c < 0)`` rate-law pairs, where the analytical entry is 0 at
probe 0 and the finite difference is the smooth slope.

The gate now records its verdict on the instance — kind, reaction or entry, probe,
analytical and finite-difference values — readable as
``last_functional_jacobian_decline()`` (``None`` after a successful attach; carried
by ``clone()``), and ``attach_functional_jacobian`` folds it into the status with
species names. The stderr line under ``BNGSIM_JAC_DEBUG=1`` stays.
"""

from __future__ import annotations

import glob
import logging
import math
from pathlib import Path

import bngsim
import pytest
from bngsim import _jacobian as J

# 0 -> B at k·A_tot²: reactant-free, so the term takes the per-species path.
# ∂f_B/∂A = 2·k·A = 2.8 at the seed and ∂f_B/∂B = 0; the pattern has one slot, (B, A).
_NET_SQUARE = """\
begin parameters
    1 k 0.7
end parameters
begin species
    1 A() 2.0
    2 B() 1.0
end species
begin functions
    1 fSq() k*A_tot*A_tot
end functions
begin reactions
    1 0 2 fSq #source_B
end reactions
begin groups
    1 A_tot 1
    2 B_tot 2
end groups
"""

# A -> B at k·A_tot: a reactant, so the term takes the per-observable path (the C++
# side applies the mass-action product rule). C is in no rate law, so (A, C) and
# (B, C) are outside the pattern.
_NET_PER_OBS = """\
begin parameters
    1 k 0.7
end parameters
begin species
    1 A() 2.0
    2 B() 1.0
    3 C() 3.0
end species
begin functions
    1 fA() k*A_tot
end functions
begin reactions
    1 1 2 fA #A_to_B
end reactions
begin groups
    1 A_tot 1
    2 B_tot 2
    3 C_tot 3
end groups
"""

# k·sqrt(A) with A = 0 at the seed: the RHS is a finite 0 there but ∂/∂A = k/(2·sqrt(A))
# is not, which the gate rightly refuses (the Newton solve would see the inf).
_NET_SQRT_AT_ZERO = """\
begin parameters
    1 k 0.7
end parameters
begin species
    1 A() 0.0
    2 B() 1.0
end species
begin functions
    1 fRoot() k*sqrt(A_tot)
end functions
begin reactions
    1 0 2 fRoot #source_B
end reactions
begin groups
    1 A_tot 1
    2 B_tot 2
end groups
"""

_MISMATCH_REASON = (
    "the finite-difference self-check found a trustworthy mismatch at probe 0 "
    "(the seed state) in ∂f[B()]/∂[A()]: analytical 5.6, finite-difference 2.8"
)
_NONFINITE_REASON = (
    "∂f[B()]/∂[A()] is non-finite (inf) at probe 0 (the seed state) while the RHS is finite"
)


def _clear_selfcheck_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run against the DEFAULT gate: no env overrides may relax or bypass it."""
    for var in (
        "BNGSIM_JAC_SELFCHECK_DENSE_MAX",
        "BNGSIM_JAC_SELFCHECK_SAMPLE",
        "BNGSIM_JAC_NO_SELFCHECK",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(params=["dense", "sparse"])
def gate(request, monkeypatch: pytest.MonkeyPatch) -> str:
    """Both self-check paths: the dense exhaustive one, and the sparse column-sampled
    one a large model gets (forced here by a zero dense ceiling)."""
    _clear_selfcheck_env(monkeypatch)
    if request.param == "sparse":
        monkeypatch.setenv("BNGSIM_JAC_SELFCHECK_DENSE_MAX", "0")
    return request.param


@pytest.fixture
def info(caplog):
    with caplog.at_level(logging.INFO, logger="bngsim"):
        yield caplog


def _decline_lines(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "bngsim"
        and r.levelno == logging.INFO
        and "analytical Jacobian" in r.getMessage()
    ]


def _model(tmp_path: Path, text: str) -> bngsim.Model:
    p = tmp_path / "m.net"
    p.write_text(text)
    return bngsim.Model.from_net(str(p))


def _derived_terms(core):
    """The derived terms exactly as attach_functional_jacobian hands them over."""
    ctx = core.functional_jacobian_context()
    func_map = dict(ctx["function_map"])
    obs_groups = {n: [(int(s), float(f)) for s, f in g] for n, g in ctx["observables"]}
    obs_idx = {n: i for i, (n, _g) in enumerate(ctx["observables"])}
    smeta = {i: (bool(a), float(v)) for i, (a, v) in enumerate(ctx["species_meta"])}
    consts = set(ctx["constant_names"])
    terms = []
    for rxn in ctx["functional_reactions"]:
        if rxn["apply_species_factor"] and rxn["reactant_idx0"]:
            t = J.build_per_observable_terms(rxn["rate_expr"], func_map, set(obs_groups), consts)
            assert t is not None, rxn["rate_expr"]
            terms.append((rxn["rxn_idx"], True, [(obs_idx[n], e) for n, e in t]))
        else:
            t = J.build_per_species_terms(rxn["rate_expr"], func_map, obs_groups, smeta, consts)
            assert t is not None, rxn["rate_expr"]
            terms.append((rxn["rxn_idx"], False, [(int(j), e) for j, e in t]))
    return terms


def _doubled(terms):
    """Every derivative 2× wrong: a mismatch the gate must judge."""
    return [(ri, po, [(j, f"2.0*({e})") for j, e in dl]) for ri, po, dl in terms]


# ─── The self-check's verdicts: the entry, the probe, both values ────────────


def test_fd_mismatch_names_the_entry_the_probe_and_both_values(tmp_path, gate):
    core = _model(tmp_path, _NET_SQUARE)._core
    assert core.last_functional_jacobian_decline() is None  # nothing has run yet
    assert core.set_functional_jacobian(_doubled(_derived_terms(core))) is False
    why = core.last_functional_jacobian_decline()
    assert why["kind"] == "fd_mismatch"
    assert why["probe"] == 0  # judged at the seed state
    assert (why["row"], why["col"]) == (1, 0)  # ∂f[B]/∂[A]
    assert why["analytical"] == pytest.approx(2 * 2.8, rel=1e-9)  # the doubled derivative
    assert why["finite_difference"] == pytest.approx(2.8, rel=1e-6)  # the slope itself
    assert J.attach_decline_reason(core) == _MISMATCH_REASON


def test_nonfinite_entry_names_the_entry_and_counts_the_rest(tmp_path, gate):
    core = _model(tmp_path, _NET_SQRT_AT_ZERO)._core
    assert core.set_functional_jacobian(_derived_terms(core)) is False
    why = core.last_functional_jacobian_decline()
    assert why["kind"] == "nonfinite_entry"
    assert why["probe"] == 0
    assert (why["row"], why["col"]) == (1, 0)
    assert math.isinf(why["analytical"]) and why["analytical"] > 0
    assert why["n_nonfinite"] == 1
    assert math.isnan(why["finite_difference"])  # no finite difference is involved
    assert J.attach_decline_reason(core) == _NONFINITE_REASON


# ─── The status and the log carry the verdict ────────────────────────────────


def test_status_names_the_mismatching_entry(tmp_path, monkeypatch, info):
    """Through the whole attach driver: a derivation doubled on its way to the gate."""
    _clear_selfcheck_env(monkeypatch)
    derive = J.build_per_species_terms

    def doubled(*args, **kwargs):
        terms = derive(*args, **kwargs)
        return None if terms is None else [(j, f"2.0*({e})") for j, e in terms]

    monkeypatch.setattr(J, "build_per_species_terms", doubled)
    m = _model(tmp_path, _NET_SQUARE)
    assert m.prepare_analytical_jacobian() is False
    assert m.analytical_jacobian_status == f"declined: {_MISMATCH_REASON}"
    assert any(_MISMATCH_REASON in line for line in _decline_lines(info))
    assert m.clone().analytical_jacobian_status == f"declined: {_MISMATCH_REASON}"


def test_status_names_the_nonfinite_entry(tmp_path, monkeypatch, info):
    _clear_selfcheck_env(monkeypatch)
    m = _model(tmp_path, _NET_SQRT_AT_ZERO)
    assert m.prepare_analytical_jacobian() is False
    assert m.analytical_jacobian_status == f"declined: {_NONFINITE_REASON}"
    assert any(_NONFINITE_REASON in line for line in _decline_lines(info))


def test_debug_stderr_line_stays(tmp_path, monkeypatch, capfd):
    _clear_selfcheck_env(monkeypatch)
    monkeypatch.setenv("BNGSIM_JAC_DEBUG", "1")
    m = _model(tmp_path, _NET_SQRT_AT_ZERO)
    assert m.prepare_analytical_jacobian() is False
    err = capfd.readouterr().err
    assert "non-finite analytical entry at (row=1, col=0)" in err, err


# ─── Every other refusal is named too ────────────────────────────────────────


def test_compile_failure_names_the_term_and_quotes_the_compiler(tmp_path):
    core = _model(tmp_path, _NET_SQUARE)._core
    terms = [(0, False, [(0, "k*(")])]
    assert core.set_functional_jacobian(terms) is False
    why = core.last_functional_jacobian_decline()
    assert why["kind"] == "compile_failed"
    assert (why["rxn_idx"], why["per_observable"], why["target_idx"]) == (0, False, 0)
    assert why["detail"]
    reason = J.attach_decline_reason(core, terms)
    assert reason.startswith(
        "the derivative of reaction 0's rate law with respect to A(), 'k*(', did not compile: "
    ), reason
    assert " ".join(why["detail"].split()) in reason


def test_per_observable_compile_failure_names_the_observable(tmp_path):
    core = _model(tmp_path, _NET_PER_OBS)._core
    terms = [(0, True, [(0, "k*(")])]
    assert core.set_functional_jacobian(terms) is False
    why = core.last_functional_jacobian_decline()
    assert why["kind"] == "compile_failed"
    assert (why["rxn_idx"], why["per_observable"], why["target_idx"]) == (0, True, 0)
    assert J.attach_decline_reason(core, terms).startswith(
        "the derivative of reaction 0's rate law with respect to A_tot, 'k*(', did not compile: "
    )


def test_entry_outside_the_pattern_is_named_on_the_per_species_path(tmp_path):
    core = _model(tmp_path, _NET_SQUARE)._core
    assert core.set_functional_jacobian([(0, False, [(1, "k")])]) is False  # ∂/∂B: no slot
    why = core.last_functional_jacobian_decline()
    assert why["kind"] == "term_outside_pattern"
    assert (why["rxn_idx"], why["row"], why["col"]) == (0, 1, 1)
    assert J.attach_decline_reason(core) == (
        "the derivative of reaction 0's rate law lands in ∂f[B()]/∂[B()], which has no slot "
        "in the Jacobian sparsity pattern"
    )


def test_entry_outside_the_pattern_is_named_on_the_per_observable_path(tmp_path):
    core = _model(tmp_path, _NET_PER_OBS)._core
    c_tot = list(core.observable_names).index("C_tot")
    assert core.set_functional_jacobian([(0, True, [(c_tot, "k")])]) is False
    why = core.last_functional_jacobian_decline()
    assert why["kind"] == "term_outside_pattern"
    assert why["per_observable"] is True
    assert why["col"] == 2 and why["row"] in (0, 1)  # (A or B, C): C is in no rate law
    assert "/∂[C()], which has no slot" in J.attach_decline_reason(core)


def test_bad_indices_are_named(tmp_path):
    core = _model(tmp_path, _NET_PER_OBS)._core
    assert core.set_functional_jacobian([(7, False, [(0, "k")])]) is False
    why = core.last_functional_jacobian_decline()
    assert (why["kind"], why["rxn_idx"]) == ("bad_reaction_index", 7)
    assert J.attach_decline_reason(core) == (
        "a derived term names reaction 7, which the model does not have"
    )
    assert core.set_functional_jacobian([(0, True, [(99, "k")])]) is False
    why = core.last_functional_jacobian_decline()
    assert (why["kind"], why["rxn_idx"], why["target_idx"]) == ("bad_observable_index", 0, 99)
    assert J.attach_decline_reason(core) == (
        "a derived term of reaction 0 names observable 99, which the model does not have"
    )


def test_a_missing_verdict_is_still_a_reason():
    """The formatter's floor, should a future refusal forget to record itself."""

    class Mute:
        species_names = ["A()"]

        def last_functional_jacobian_decline(self):
            return None

    assert J.attach_decline_reason(Mute()).startswith(
        "the C++ attach declined the derived terms without recording why"
    )


# ─── The record describes the last call only, and a clone carries it ─────────


def test_a_successful_attach_clears_the_verdict_and_a_clone_carries_it(tmp_path, monkeypatch):
    _clear_selfcheck_env(monkeypatch)
    core = _model(tmp_path, _NET_SQUARE)._core
    terms = _derived_terms(core)
    assert core.set_functional_jacobian(_doubled(terms)) is False
    why = core.last_functional_jacobian_decline()
    assert why["kind"] == "fd_mismatch"
    assert core.clone().last_functional_jacobian_decline() == why
    assert core.set_functional_jacobian(terms) is True
    assert core.last_functional_jacobian_decline() is None
    assert core.clone().last_functional_jacobian_decline() is None


# ─── The corpus model of the issue ───────────────────────────────────────────

_LORENZ = glob.glob("benchmarks/suites/ode_fullnet/nets/*ph_lorenz_attractor.bngl.net")


@pytest.mark.skipif(not _LORENZ, reason="benchmark ode_fullnet corpus not present")
def test_lorenz_seed_on_surface_reads_from_the_status_alone(monkeypatch):
    """The seed-on-surface class: each signed rate is split into an ``if(c > 0, c, 0)``
    and an ``if(c < 0, -c, 0)`` reaction, and the seed sits on ``c = 0`` for all three,
    so the analytical entry is 0 at probe 0 while the finite difference is the smooth
    slope. That used to be one generic bucket; the status now says so itself."""
    _clear_selfcheck_env(monkeypatch)
    m = bngsim.Model.from_net(_LORENZ[0])
    assert m.prepare_analytical_jacobian() is False
    assert m.analytical_jacobian_status == (
        "declined: the finite-difference self-check found a trustworthy mismatch at probe 0 "
        "(the seed state) in ∂f[LX()]/∂[LX()]: analytical 0, finite-difference -10"
    )
