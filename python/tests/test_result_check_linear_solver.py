"""``Result.rerun(linear_solver=...)`` and ``check("linear_solver")`` (issue #1888).

``rerun()`` takes the linear solver as ``"dense"``, ``"klu"`` or ``"auto"``, set
through the Simulator's two force flags, so ``check()`` can make a run again
changing that alone: a dense run on KLU, a KLU run dense. The numbers are held
to ``decay``'s closed form on each solver; a ``diff`` is made to order (bngsim's
two solvers agree on every bundled example), and a ``skip`` names why KLU did
not run.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from bngsim import Result, Simulator, UsageError, _check, _rerun, examples
from bngsim._simulator import _HAS_KLU

needs_klu = pytest.mark.skipif(not _HAS_KLU, reason="this build has no KLU")


def _decay(**sim) -> Result:
    return Simulator(examples.load("decay"), method="ode", **sim).run(
        t_span=(0, 5), n_points=11, rtol=1e-10, atol=1e-12
    )


def _solver(r: Result) -> tuple[str, str]:
    b = r.record["backend"]
    return b["linear_solver"], b["linear_solver_requested"]


def _closed_form(r: Result) -> None:
    """``A(t) = A0·exp(-k·t)``, at the run's own parameters."""
    p = r.record["run"]["parameters"]
    expect = p["A0"] * np.exp(-p["k"] * np.asarray(r.time))
    np.testing.assert_allclose(np.asarray(r.species)[:, 0], expect, rtol=1e-7, atol=0)


# ─── rerun(linear_solver=...) ────────────────────────────────────────────────


@needs_klu
def test_each_solver_runs_and_holds_the_closed_form() -> None:
    r = _decay()
    assert _solver(r) == ("dense", "auto")
    _closed_form(r)
    klu = r.rerun(linear_solver="klu")
    assert _solver(klu) == ("klu", "sparse")
    assert klu.record["run"]["rerun_of"]["changed"] == {"linear_solver": "klu"}
    _closed_form(klu)
    dense = klu.rerun(linear_solver="dense")
    assert _solver(dense) == ("dense", "dense")
    _closed_form(dense)
    # "auto" takes a forced solver off: the Simulator's own rule, dense here.
    assert _solver(klu.rerun(linear_solver="auto")) == ("dense", "auto")
    # Left out, a forced solver is kept.
    assert _solver(klu.rerun()) == ("klu", "sparse")


@needs_klu
def test_a_changed_solver_is_no_different_backend_notice() -> None:
    again = _decay().rerun(linear_solver="klu")
    assert "rerun.different_backend" not in [n["code"] for n in again.notices]


@needs_klu
def test_a_stochastic_run_rerun_as_ode_takes_a_solver() -> None:
    r = Simulator(examples.load("decay"), method="ssa").run(t_span=(0, 5), n_points=11, seed=1)
    again = r.rerun(method="ode", linear_solver="klu")
    assert _solver(again) == ("klu", "sparse")


@pytest.mark.parametrize("value", ["sparse", "KLU", "", 1, True])
def test_a_solver_it_does_not_know_is_refused(value) -> None:
    with pytest.raises(UsageError) as e:
        _decay().rerun(linear_solver=value)
    assert e.value.code == "usage.invalid_argument"
    assert e.value.details["accepted"] == ["auto", "dense", "klu"]


def test_none_is_refused_as_changing_nothing() -> None:
    with pytest.raises(UsageError, match="changes nothing"):
        _decay().rerun(linear_solver=None)


def test_a_solver_for_a_method_that_has_none_is_refused() -> None:
    r = _decay()
    for method in ("ssa", "SSA"):
        with pytest.raises(UsageError) as e:
            r.rerun(method=method, linear_solver="dense")
        assert e.value.details == {"arguments": ["linear_solver"], "method": "ssa"}
    s = Simulator(examples.load("decay"), method="ssa").run(t_span=(0, 5), n_points=3, seed=1)
    with pytest.raises(UsageError, match="'ssa'"):
        s.rerun(linear_solver="dense")


# ─── check("linear_solver") ──────────────────────────────────────────────────


@needs_klu
def test_the_check_runs_the_other_solver_each_way() -> None:
    dense = _decay()
    (c,) = dense.check("linear_solver")["checks"]
    assert (c["name"], c["status"], c["changed"]) == (
        "linear_solver",
        "pass",
        {"linear_solver": "klu"},
    )
    klu = _decay(force_sparse_linear_solver=True)
    (c,) = klu.check("linear_solver")["checks"]
    assert (c["status"], c["changed"]) == ("pass", {"linear_solver": "dense"})


def _record_saying(monkeypatch, **backend) -> None:
    """Make every result's record say *backend* of the code that ran."""
    real = Result.record.fget

    def record(self):
        rec = copy.deepcopy(real(self))
        rec["backend"].update(backend)
        return rec

    monkeypatch.setattr(Result, "record", property(record))


@needs_klu
def test_a_run_on_the_blas_dense_solver_is_checked_against_klu(monkeypatch) -> None:
    r = _decay()
    _record_saying(monkeypatch, linear_solver="lapack_dense")
    (c,) = r.check("linear_solver")["checks"]
    # The rerun's record says "lapack_dense" too, so KLU reads as not run.
    assert c["changed"] == {"linear_solver": "klu"} and c["status"] == "skip"


def _perturbing(monkeypatch, factor: float) -> None:
    real = Result.rerun

    def rerun(self, **changes):
        out = real(self, **changes)
        if "linear_solver" in changes:
            out = copy.copy(out)
            out._species = np.asarray(out._species, dtype=float) * factor
        return out

    monkeypatch.setattr(Result, "rerun", rerun)


@needs_klu
def test_a_solver_that_disagrees_is_a_diff(monkeypatch) -> None:
    r = _decay()
    _perturbing(monkeypatch, 1.05)
    rep = r.check(["tolerance", "linear_solver"])
    c = {x["name"]: x for x in rep["checks"]}
    assert c["tolerance"]["status"] == "pass"
    assert c["linear_solver"]["status"] == "diff"
    assert c["linear_solver"]["worst_column"] == "A()"
    assert c["linear_solver"]["value"] == pytest.approx(0.05, rel=1e-6)
    assert rep["ok"] is False


@needs_klu
def test_a_disagreement_within_the_allowance_passes(monkeypatch) -> None:
    r = _decay()
    _perturbing(monkeypatch, 1 + 1e-9)
    (c,) = r.check("linear_solver")["checks"]
    assert c["status"] == "pass"


@pytest.mark.parametrize("has_klu", [False, True])
def test_a_run_klu_cannot_make_is_a_skip_that_says_why(monkeypatch, has_klu: bool) -> None:
    # "klu" made to run the Simulator's own rule, as a build or model KLU
    # cannot take does: dense.
    monkeypatch.setitem(_rerun.LINEAR_SOLVERS, "klu", (False, False))
    monkeypatch.setattr("bngsim._simulator._HAS_KLU", has_klu)
    rep = _decay().check("linear_solver")
    (c,) = rep["checks"]
    assert c["status"] == "skip", c
    assert c["changed"] == {"linear_solver": "klu"}
    assert "ran 'dense'" in c["note"]
    reason = "built without KLU" if not has_klu else "sparse Jacobian pattern"
    assert reason in c["note"]
    # A skip alone passes nothing; beside a check that passes it is no finding.
    assert rep["ok"] is False
    assert _decay().check(["tolerance", "linear_solver"])["ok"] is True


def test_a_jax_jacobian_is_named_as_why() -> None:
    note = _check._no_klu(None, {"backend": {"jacobian": "jax"}})  # type: ignore[arg-type]
    assert note == (
        "a JAX Jacobian runs dense only" if _HAS_KLU else "this bngsim was built without KLU"
    )


def test_a_stochastic_run_and_a_record_without_a_solver_are_unsupported(monkeypatch) -> None:
    s = Simulator(examples.load("decay"), method="ssa").run(t_span=(0, 5), n_points=3, seed=1)
    (c,) = s.check("linear_solver")["checks"]
    assert c["status"] == "unsupported" and "stochastic" in c["note"]
    r = _decay()
    _record_saying(monkeypatch, linear_solver=None)
    (c,) = r.check("linear_solver")["checks"]
    assert c["status"] == "unsupported" and "which solver" in c["note"]


def test_all_runs_it_after_the_jacobian() -> None:
    names = [c["name"] for c in _decay().check("all")["checks"]]
    assert names[:4] == ["tolerance", "grid", "jacobian", "linear_solver"]
    assert "linear_solver" not in [c["name"] for c in _decay().check()["checks"]]
