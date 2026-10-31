"""Shape models through every path that should give the same answer (issue #1257).

``parity_checks/differential/_differential.py`` runs one model through the
paths that should agree (a sensitivity run's state and a plain run's, analytic
and FD Jacobians, KLU and dense, compiled and interpreted, cc and MIR) and three
rules for bugs whose answers are right (a sensitivity run refused, a sparsity
pattern that misses entries, a backend other than the one asked for). The nightly
runs it over the corpus (``parity_checks/differential``); this runs it over the
shapes the corpus rarely holds, on every pull request:

* triangle pulses over the clock written with ``max``/``abs``, ``clamp`` and
  ``if()``, and a ``clamp`` ramp, each wide (w = 0.5) and narrow (w = 0.01):
  only the narrow ones, at several tolerances, found #1227;
* an equality the state sits on, ``if(Bobs==0, kb, 0)`` (#1228, #1254);
* a derived rate constant over ``log10`` (#1214, fixed by #1224);
* a state trigger that crosses exactly at a sample time (#1182);
* a compartment an assignment rule moves, so its species are diluted;
* ``rateOf`` across compartments (#1266);
* a daily pulse whose width an event sets, between pulses and inside one
  (#1244: the grid check, 8x the output points read at the base's own, found
  it 7.5x the trajectory's peak off before #1263).

Each check is asked to pass or to say why it does not apply. A known bug that is
still open is a strict xfail naming its issue, so its fix turns it into a
failure to be removed. Each check is also shown to fail on a defect it is there
to see (the negative controls below), so a check that cannot fail does not
read as a pass.

The locale check is the nightly's: it needs a comma-decimal locale, which the
nightly generates and this suite's runners do not have.
"""

from __future__ import annotations

import atexit
import functools
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import bngsim
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source_root import bngsim_source_root  # noqa: E402

_ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "parity_checks" / "differential"))

import _differential as D  # noqa: E402
import _transforms as X  # noqa: E402

antimony = pytest.importorskip("antimony")

# The ODE checks: the SSA arm (#1310) runs on SSA jobs, which the shapes are not.
CHECKS = tuple(c for c in D.CHECKS if c != "locale" and c not in D.SSA_CHECKS + D.NF_CHECKS)
_DIR = Path(tempfile.mkdtemp(prefix="bngsim-differential-shapes-"))
atexit.register(shutil.rmtree, _DIR, ignore_errors=True)


def _net(name, params, species, functions, reactions, groups) -> Path:
    path = _DIR / f"{name}.net"
    path.write_text(
        f"begin parameters\n{params}\nend parameters\nbegin species\n{species}\nend species\n"
        f"begin functions\n{functions}\nend functions\n"
        f"begin reactions\n{reactions}\nend reactions\nbegin groups\n{groups}\nend groups\n"
    )
    return path


def _sbml(name, text) -> Path:
    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString(f"model m\n{text}\nend") >= 0, antimony.getLastError()
    path = _DIR / f"{name}.xml"
    path.write_text(antimony.getSBMLString("m"))
    return path


_PULSE = " 1 kx 1\n 2 k_clock 2\n 3 c 5\n 4 w {w}"
_LAWS = {
    "triangle-maxabs": "kx*(1/w)*max(0,1-abs(Tobs-c)/(w/2))",
    "triangle-clamp": "kx*(1/w)*(clamp(0,(Tobs-c+w/2)/(w/2),1)-clamp(0,(Tobs-c)/(w/2),1))",
    "triangle-if": "kx*(1/w)*if(Tobs<c-w/2,0,if(Tobs<c,(Tobs-c+w/2)/(w/2),"
    "if(Tobs<c+w/2,(c+w/2-Tobs)/(w/2),0)))",
    "ramp-clamp": "kx*clamp(0,(Tobs-c)/w,1)",
}


def _shapes() -> dict[str, D.Job]:
    jobs = {}
    for name, law in _LAWS.items():
        for suffix, w in (("", "0.5"), ("-narrow", "0.01")):
            path = _net(
                name + suffix,
                _PULSE.format(w=w),
                " 1 T() 0\n 2 X() 10\n 3 Y() 0",
                f" 1 f() {law}",
                " 1 0 1 k_clock\n 2 2 3 f",
                " 1 Tobs 1",
            )
            jobs[name + suffix] = D.Job(name + suffix, path, 0.0, 10.0, 21)
    jobs["equality-dwell"] = D.Job(
        "equality-dwell",
        _net(
            "equality-dwell",
            " 1 a 0.5\n 2 q 2.0\n 3 kb 1.0\n 4 kp 1.0",
            " 1 A() 10\n 2 B() 0\n 3 Y() 0",
            " 1 fB() kp*if(q-Aobs>0,q-Aobs,0)\n 2 fY() if(Bobs==0,kb,0)",
            " 1 1 0 a\n 2 0 2 fB\n 3 0 3 fY",
            " 1 Aobs 1\n 2 Bobs 2",
        ),
        0.0,
        10.0,
        21,
    )
    jobs["derived-log10"] = D.Job(
        "derived-log10",
        _net(
            "derived-log10",
            " 1 p 100.0\n 2 k log10(p)",
            " 1 A() 1.0\n 2 B() 0",
            "",
            " 1 1 2 k",
            " 1 Aobs 1",
        ),
        0.0,
        10.0,
        21,
    )
    sbml = {
        # A = 10·exp(-t·ln2/5) is 5 at t = 5, a sample time
        "event-at-a-sample": (
            "A = 10; k = 0.1386294361119891; J: A -> ; k*A; B = 0; E: at (A < 5): B = B + 1;",
            10.0,
        ),
        "moving-compartment": (
            "compartment cv; cv := 1 + 0.1*time; species X in cv = 4; k = 0.25; J: X -> ; k*X*cv;",
            10.0,
        ),
        "rateof-across": (
            "compartment C1 = 2; compartment C2 = 3; k = 0.3; g = 0.7; species A in C1 = 1;"
            " species B in C2 = 0; species S in C1 = 0; J: A -> B; k*A*C1; S' = g*rateOf(B);",
            20.0,
        ),
        "pulse-width-event": (
            "compartment C = 1; species X in C; X = 0; w = 1; a = 1;"
            " J0: -> X; a*piecewise(1, (time - 24*floor(time/24)) < w, 0);"
            " E1: at (time >= 50): w = 2;",
            230.0,
        ),
        # the same, the width set inside a pulse (t = 72.5, in the one from 72)
        "pulse-width-event-midpulse": (
            "compartment C = 1; species X in C; X = 0; w = 1; a = 1;"
            " J0: -> X; a*piecewise(1, (time - 24*floor(time/24)) < w, 0);"
            " E1: at (time >= 72.5): w = 2;",
            230.0,
        ),
    }
    for name, (text, t_end) in sbml.items():
        jobs[name] = D.Job(name, _sbml(name, text), 0.0, t_end, 21)
    return jobs


SHAPES = _shapes()

# (shape, check) -> what is expected instead of a pass: a declared refusal is
# one by name; an open bug is a strict xfail on its issue.
REFUSED = {
    # #1240: refused by name, the state on the equality's surface
    ("equality-dwell", "sens_runs"): "unsupported",
    # #710: steady_state() refuses a model with an event or one that reads the time
    **{
        (shape, check): "unsupported"
        for shape in (
            "event-at-a-sample",
            "moving-compartment",
            "pulse-width-event",
            "pulse-width-event-midpulse",
        )
        for check in D.STEADY_METHODS
    },
}
# Empty since #1254 was fixed (KLU's colored difference quotient steps on the
# error weights); a newly found bug goes here.
OPEN: dict[tuple[str, str], str] = {}


@functools.cache
def _verdicts(shape: str) -> dict[str, D.Verdict]:
    return D.check_all(SHAPES[shape], None, CHECKS)


def _cases():
    for shape in SHAPES:
        for check in CHECKS:
            marks = ()
            if (shape, check) in OPEN:
                # Only where the path runs: a build without KLU skips the KLU
                # check, which is right, and is no pass of a KLU bug's (the MIR
                # leg builds without it).
                marks = pytest.mark.xfail(
                    condition=check != "klu_vs_dense" or bool(bngsim._bngsim_core.HAS_KLU),
                    strict=True,
                    reason=OPEN[shape, check],
                )
            yield pytest.param(shape, check, id=f"{shape}-{check}", marks=marks)


@pytest.mark.parametrize(("shape", "check"), list(_cases()))
def test_each_path_gives_the_same_answer(shape, check):
    v = _verdicts(shape)[check]
    expected = REFUSED.get((shape, check))
    if expected is not None:
        assert v.status == expected, v
    else:
        assert v.status in ("pass", "skip"), f"{v.status}: {v.note}"


@pytest.mark.parametrize("shape", [s for s in SHAPES if (s, "sens_runs") not in REFUSED])
def test_the_sensitivity_rhs_is_analytic(shape):
    """A slow fallback with right answers is the #1214 class: the answers agree,
    so only the path it took shows it. derived-log10 declined until #1224."""
    v = _verdicts(shape)["sens_runs"]
    assert v.extra.get("sens_rhs") == "analytic", v.extra


# ── Negative controls: each check fails on the defect it is there to see ──


def test_a_path_off_by_more_than_the_bound_is_a_finding():
    base = D.Base(SHAPES["triangle-maxabs"])
    assert base.compare(base.x).status == "pass"
    off = base.x * (1 + 10 * base.limit)
    v = base.compare(off)
    assert v.status == "diff" and v.value > base.limit, v


def test_a_path_within_atol_is_not_a_finding():
    """A cell is judged past an absolute floor of the run's atol, scaled as the
    comparison scales it: a species at 1e-14 in a model of 10 is not one."""
    base = D.Base(SHAPES["triangle-maxabs"])
    x = base.x.copy()
    x[:, -1] += 0.5 * D.PATH_RTOL_FACTOR * base.job.atol
    assert base.compare(x).status == "pass"


def test_a_trajectory_smaller_than_atol_is_judged_past_it():
    """A nanomolar model's every cell is within atol of another path's: no cell
    of a run held to atol is asked to agree closer. Judged on the peak alone,
    a difference of 5e-11 in a state of 1e-9 read as 5%."""
    ref = np.full((3, 2), 1e-9)
    assert D.gap(ref + 5e-11, ref, 1e-10) == 0.0
    assert D.gap(ref + 5e-10, ref, 1e-10) > 0.0


def test_a_shape_or_a_non_finite_cell_is_a_finding():
    ref = np.ones((3, 2))
    assert D.gap(np.ones((3, 3)), ref, 1e-10) == float("inf")
    bad = ref.copy()
    bad[1, 1] = np.nan
    assert D.gap(bad, ref, 1e-10) == float("inf")


class _FakeCore:
    def __init__(self, pattern):
        self.jacobian_sparsity = pattern


class _FakeModel:
    """dA/dt = -k·A·B, dB/dt = -A: every entry but d(dB/dt)/dB is nonzero."""

    def __init__(self, pattern):
        self._core = _FakeCore(pattern)

    @staticmethod
    def rhs(y, t=0.0):
        a, b = y
        return np.array([-2.0 * a * b, -a])


def _csc(dense):
    dense = np.asarray(dense, dtype=bool)
    n = dense.shape[0]
    cols, rows = [0], []
    for j in range(n):
        rows += [i for i in range(n) if dense[i, j]]
        cols.append(len(rows))
    return {"n": n, "col_ptrs": np.array(cols), "row_indices": np.array(rows)}


def test_a_sparsity_pattern_that_misses_an_entry_is_a_finding():
    y = [np.array([1.0, 2.0])]
    full = _FakeModel(_csc([[1, 1], [1, 0]]))
    assert D.sparsity_misses(full, y) == []
    missing = _FakeModel(_csc([[1, 0], [1, 0]]))  # no slot for d(dA/dt)/dB
    assert D.sparsity_misses(missing, y) == [(0, 1)]


def test_an_entry_the_differences_cannot_resolve_is_not_asked_of_the_pattern():
    """An entry under the floor of the largest contribution is no entry."""

    class Tiny(_FakeModel):
        @staticmethod
        def rhs(y, t=0.0):
            a, b = y
            return np.array([-a + 1e-14 * b, -a])

    assert D.sparsity_misses(Tiny(_csc([[1, 0], [1, 0]])), [np.array([1.0, 2.0])]) == []


class _JacModel(_FakeModel):
    """_FakeModel with an analytic Jacobian, ``wrong`` written over its entries."""

    def __init__(self, wrong=None, source="analytical"):
        super().__init__(_csc([[1, 1], [1, 0]]))
        self.wrong, self.source = wrong or {}, source

    def jacobian(self, y, t=0.0):
        from bngsim._evaluators import JacobianMatrix

        a, b = y
        J = np.array([[-2.0 * b, -2.0 * a], [-1.0, 0.0]])
        for (i, j), v in self.wrong.items():
            J[i, j] = v
        return JacobianMatrix(J, self.source)


def test_a_jacobian_with_one_wrong_entry_is_a_finding():
    """#1308's control: d(dA/dt)/dB is -2a = -2; written as -2.2 (10% off), the
    check names that entry and no other."""
    y = [np.array([1.0, 2.0])]
    assert D.jacobian_misfits(_JacModel(), y) == []
    found = D.jacobian_misfits(_JacModel({(0, 1): -2.2}), y)
    assert [m["entry"] for m in found] == [(0, 1)]
    assert found[0]["off"] == pytest.approx(0.2 / 2.2, rel=1e-3)


def test_a_one_percent_error_is_a_finding():
    found = D.jacobian_misfits(_JacModel({(1, 0): -1.01}), [np.array([1.0, 2.0])])
    assert [m["entry"] for m in found] == [(1, 0)]


def test_a_spurious_entry_where_the_rhs_reads_nothing_is_a_finding():
    """d(dB/dt)/dB is 0 in the RHS; an analytic -0.5 there is wrong."""
    found = D.jacobian_misfits(_JacModel({(1, 1): -0.5}), [np.array([1.0, 2.0])])
    assert [m["entry"] for m in found] == [(1, 1)]


def test_a_non_finite_analytic_entry_is_a_finding():
    found = D.jacobian_misfits(_JacModel({(0, 0): float("nan")}), [np.array([1.0, 2.0])])
    assert [m["entry"] for m in found] == [(0, 0)] and found[0]["off"] == float("inf")


def test_a_differenced_jacobian_is_not_judged():
    """Model.jacobian differences the RHS where the closed form is incomplete:
    nothing analytic to judge."""
    y = [np.array([1.0, 2.0])]
    assert D.jacobian_misfits(_JacModel(source="finite-difference"), y) is None


def test_a_model_whose_difference_quotient_jacobian_refuses_is_skipped(monkeypatch):
    """Model.jacobian() refuses a difference quotient whose step decides an
    entry (#1002), where it returned one: a model without the closed form is
    SKIP, as the baseline holds it, not an EXCEPTION. BIOMD0000000053 and six
    more BioModels read EXCEPTION once main refused."""
    base = _base("derived-log10")

    def refuses(self, y, t=0.0, **kw):
        raise bngsim.SimulationError(
            "Model.jacobian() does not return this difference quotient (issue #1002)"
        )

    monkeypatch.setattr(bngsim.Model, "prepare_analytical_jacobian", lambda self: False)
    monkeypatch.setattr(bngsim.Model, "jacobian", refuses)
    v = D.check_jacobian_values(base)
    assert v.status == "skip" and "no analytic Jacobian" in v.note, v


def test_rounding_in_a_large_rhs_is_not_a_finding():
    """blbr's false alarm: f is 1e11, so a difference over a small step rounds by
    eps·|f|/h. At a step of 1e-6 its differences agreed on 2.0345 at two steps
    where the entry is 2. The gaps between neighbouring steps bound that
    rounding, so it does not read as a wrong entry even on a ladder of small
    steps only, while on the check's own ladder a 10% error still reads as one.

    (Rounding is the limit of what this check can judge: an entry no step
    resolves is not judged at all.)"""

    class Big(_JacModel):
        @staticmethod
        def rhs(y, t=0.0):
            a, b = y
            return np.array([1e9 - 2.0 * a * b, -a])

    y = [np.array([1.0, 2.0])]
    assert D.jacobian_misfits(Big(), y) == []
    assert D.jacobian_misfits(Big(), y, steps=(1e-6, 5e-7, 2.5e-7)) == []
    assert [m["entry"] for m in D.jacobian_misfits(Big({(0, 1): -2.2}), y)] == [(0, 1)]


def test_an_entry_on_a_kink_is_not_judged():
    """The #368 trap: dA/dt = -|b - 2| at b = 2. Every central difference reads
    0, the mean of the slopes -1 and +1 either side; the analytic entry is one
    side. Neither is the derivative, so the entry is not judged."""
    from bngsim._evaluators import JacobianMatrix

    class Kink(_FakeModel):
        @staticmethod
        def rhs(y, t=0.0):
            return np.array([-abs(y[1] - 2.0), -y[0]])

        def jacobian(self, y, t=0.0):
            return JacobianMatrix(np.array([[0.0, -1.0], [-1.0, 0.0]]), "analytical")

    assert D.jacobian_misfits(Kink(_csc([[0, 1], [1, 0]])), [np.array([1.0, 2.0])]) == []


def test_a_feature_narrower_than_the_coarse_steps_is_not_a_finding():
    """BIOMD0000000498's shape: dA/dt = 5 - 2b + 1e-9·tanh(b/1e-9) at b = 0. Its
    slope at the point is -1, the analytic entry; at steps far over 1e-9 the
    differences settle, step after step, on the secant -2, and with the
    constant's rounding at the fine steps that plateau is the steps' tightest.
    A finding needs the analytic entry contradicted at every step that
    resolves it, and the fine steps agree with it; judged at the tightest step
    alone, this read as a wrong entry.

    (A feature narrower still than any step rounding lets the ladder resolve
    is past what the check can tell from a wrong entry: at 1e-10 under a
    constant of 10 it reads as one.)"""
    from bngsim._evaluators import JacobianMatrix

    class Narrow(_FakeModel):
        @staticmethod
        def rhs(y, t=0.0):
            return np.array([5.0 - 2.0 * y[1] + 1e-9 * np.tanh(y[1] / 1e-9), -y[0]])

        def jacobian(self, y, t=0.0):
            return JacobianMatrix(np.array([[0.0, -1.0], [-1.0, 0.0]]), "analytical")

    y = [np.array([1.0, 0.0])]
    assert D.jacobian_misfits(Narrow(_csc([[0, 1], [1, 0]])), y) == []
    # and with the analytic entry the coarse secant's -2, which no fine step
    # bears out, it is no finding either: the steps disagree with each other.
    wrong = Narrow(_csc([[0, 1], [1, 0]]))
    wrong.jacobian = lambda yy, t=0.0: JacobianMatrix(
        np.array([[0.0, -3.0], [-1.0, 0.0]]), "analytical"
    )
    assert [m["entry"] for m in D.jacobian_misfits(wrong, y)] == [(0, 1)]


def test_a_slope_rounded_away_at_the_fine_steps_is_not_a_finding():
    """BIOMD0000000393's shape (#533): dA/dt = (N/2)(sqrt(1 + 4b/N) - 1)
    + 100(b - 1)^3 - 0.3b at b = 1, with 4b/N = 1e-10. Under a step of about
    2e-6 of b, 1 + 4b/N rounds to what it was, so the root's slope of 1 is
    lost and the fine steps agree, step after step, on -0.3; the analytic
    entry is 0.7. The coarse steps, a few percent apart from each other for
    the cubic, read 0.7: too noisy to resolve the entry, which the fine steps
    seem to, but enough to bear the analytic entry out, so it is no finding.

    (Rounding lost deeper inside a law than the coarse steps can speak to is
    past what the check can tell from a wrong entry: at 4b/N = 1e-11 with no
    cubic it reads as one.)"""
    from bngsim._evaluators import JacobianMatrix

    N = 4.0 / 1e-10

    class Plateau(_FakeModel):
        @staticmethod
        def rhs(y, t=0.0):
            b = y[1]
            root = 0.5 * N * (np.sqrt(1.0 + 4.0 * b / N) - 1.0)
            return np.array([root + 100.0 * (b - 1.0) ** 3 - 0.3 * b, -y[0]])

        def jacobian(self, y, t=0.0):
            slope = 1.0 / np.sqrt(1.0 + 4.0 * y[1] / N) - 0.3
            return JacobianMatrix(np.array([[0.0, slope], [-1.0, 0.0]]), "analytical")

    assert D.jacobian_misfits(Plateau(_csc([[0, 1], [1, 0]])), [np.array([1.0, 1.0])]) == []


def test_a_column_of_noise_is_not_judged_against_itself():
    """mm_tqssa_stiff's false alarm: a column whose analytic entries are 2e-19
    and whose differences are 0, beside entries of order 1. On its own peak it
    reads 100% off; against the matrix it is noise."""
    from bngsim._evaluators import JacobianMatrix

    class Noise(_FakeModel):
        """dA/dt = 1 - a, dB/dt = 0, at its steady state a = 1: nothing reads
        B, so column B is 0, and the analytic side carries 2e-19 there. f is 0
        at the state, so the differences round to nothing either, and only the
        matrix's own scale says 2e-19 is noise."""

        @staticmethod
        def rhs(y, t=0.0):
            return np.array([1.0 - y[0], 0.0])

        def jacobian(self, y, t=0.0):
            return JacobianMatrix(np.array([[-1.0, 2e-19], [0.0, 0.0]]), "analytical")

    assert D.jacobian_misfits(Noise(_csc([[1, 1], [0, 0]])), [np.array([1.0, 2.0])]) == []


def test_jacobian_values_on_a_real_model_with_a_wrong_entry_is_a_diff(monkeypatch):
    """The check end to end: a real shape's analytic Jacobian, its largest entry
    10% off. derived-log10's entries are of order 2 at every state; the
    triangle pulses' are all 0 (their rates read the clock only), where a
    scaled entry changes nothing."""
    from bngsim._evaluators import JacobianMatrix

    real = bngsim.Model.jacobian

    def wrong(self, y, t=0.0, **kw):
        J = real(self, y, t, **kw)
        out = np.array(J, dtype=float)
        k = np.unravel_index(np.argmax(np.abs(out)), out.shape)
        assert out[k] != 0.0, "the corruption must change the matrix"
        out[k] *= 1.1
        return JacobianMatrix(out, J.source)

    base = _base("derived-log10")
    assert D.check_jacobian_values(base).status == "pass"
    monkeypatch.setattr(bngsim.Model, "jacobian", wrong)
    v = D.check_jacobian_values(base)
    assert v.status == "diff", v


# ── Metamorphic transforms (#1305) ──
#
# Every SBML shape above has an event, the clock or a rateOf, so none is one
# rescale_time applies to; this is. A + B -> P, P -> A: every species moves,
# and a rate rule (Z) that R2 reads, which the rescale must scale too.
_AUTONOMOUS = D.Job(
    "autonomous",
    _sbml(
        "autonomous",
        "compartment C = 1; species A in C, B in C, P in C; A = 10; B = 4; P = 0;"
        " k1 = 0.3; k2 = 0.05; Z = 1; Z' = -0.1*Z;"
        " R1: A + B -> P; k1*A*B; R2: P -> A; k2*P*(2 - Z);",
    ),
    0.0,
    20.0,
    21,
)


@functools.cache
def _job_base(name):
    return D.Base(_AUTONOMOUS if name == "autonomous" else SHAPES[name])


@pytest.mark.parametrize("kind", X.TRANSFORMS)
def test_an_autonomous_sbml_model_keeps_its_answer_under_each_transform(kind):
    v = D.check_transform(_job_base("autonomous"), kind)
    assert v.status == "pass", v


def _misapplied(monkeypatch, fix):
    """Each transform as written, then ``fix`` applied to what it made: the
    transform done wrong, which its check is there to see."""
    real = X.transform

    def wrong(kind, path, workdir):
        tr = real(kind, path, workdir)
        fix(tr)
        return tr

    monkeypatch.setattr(D, "transform", wrong)


def _swap_first_and_last_initial_values(tr):
    """The data left in the old order under the new labels: the first and the
    last species' initial values exchanged in the transformed file (in both
    models those differ; equality-dwell's first two are both 0)."""
    if tr.path.suffix == ".net":
        net = X._Net.parse(tr.path.read_text())
        rows = net.entries("species")
        (f0, c0), (f1, c1) = rows[0], rows[-1]
        rows[0], rows[-1] = ([*f0[:2], f1[2], *f0[3:]], c0), ([*f1[:2], f0[2], *f1[3:]], c1)
        net.set_entries("species", rows)
        tr.path.write_text(net.text())
        return
    import libsbml

    doc = libsbml.readSBMLFromFile(str(tr.path))
    m = doc.getModel()
    s0, s1 = m.getSpecies(0), m.getSpecies(m.getNumSpecies() - 1)
    v0, v1 = s0.getInitialConcentration(), s1.getInitialConcentration()
    s0.setInitialConcentration(v1)
    s1.setInitialConcentration(v0)
    libsbml.writeSBMLToFile(doc, str(tr.path))


# equality-dwell is a .net whose three species all move; autonomous an SBML.
MISAPPLIED_ON = ("equality-dwell", "autonomous")


@pytest.mark.parametrize("model", MISAPPLIED_ON)
def test_a_reorder_that_leaves_the_values_behind_is_a_finding(monkeypatch, model):
    _misapplied(monkeypatch, _swap_first_and_last_initial_values)
    v = D.check_transform(_job_base(model), "reorder")
    assert v.status == "diff", v


@pytest.mark.parametrize("model", MISAPPLIED_ON)
def test_a_rename_that_crosses_two_names_is_a_finding(monkeypatch, model):
    def cross(tr):
        a, b = list(tr.names)[:2]
        tr.names[a], tr.names[b] = tr.names[b], tr.names[a]

    _misapplied(monkeypatch, cross)
    v = D.check_transform(_job_base(model), "rename")
    assert v.status == "diff", v


@pytest.mark.parametrize("model", MISAPPLIED_ON)
def test_a_split_that_keeps_the_whole_rate_on_each_half_is_a_finding(monkeypatch, model):
    monkeypatch.setattr(X, "SPLIT_SHARE", 1.0)
    v = D.check_transform(_job_base(model), "split")
    assert v.status == "diff", v


@pytest.mark.parametrize("model", MISAPPLIED_ON)
def test_a_rescale_of_the_rates_and_not_the_window_is_a_finding(monkeypatch, model):
    _misapplied(monkeypatch, lambda tr: setattr(tr, "time_scale", 1.0))
    v = D.check_transform(_job_base(model), "rescale_time")
    assert v.status == "diff", v


def test_a_transform_that_does_not_apply_is_skipped_by_name():
    """An event's time is not a rate: rescale_time does not keep the answer of
    a model with one, so the row says so rather than judging."""
    v = D.check_transform(_job_base("pulse-width-event"), "rescale_time")
    assert v.status == "skip" and "events" in v.note, v


def _job_of(name, path, t_end=10.0):
    return D.Base(D.Job(name, path, 0.0, t_end, 21))


def test_a_rate_rule_that_reads_a_reaction_rate_is_not_rescaled():
    """A reaction's id in math is its rate, which the rescale scales already:
    BIOMD0000000556's rate rules read their reactions', and scaled again came
    out 9x. Refused by name instead (and RoadRunner agrees the naive rescale is
    wrong: it is the transform, not bngsim)."""
    path = _sbml(
        "reads_a_reaction",
        "compartment C = 1; species A in C; A = 10; Z = 0; k = 0.3; J1: A -> ; k*A; Z' = J1;",
    )
    v = D.check_transform(_job_of("reads_a_reaction", path), "rescale_time")
    assert v.status == "skip" and "reaction's rate" in v.note, v


_KEYWORD_NET = """begin parameters
    1 kcat 2.0
    2 Km   5.0
    3 E0   1.0
    4 kd   if((kcat>=1)&&(kcat<10), 0.1, 0.2)
end parameters
begin species
    1 E() E0
    2 S() 20
    3 P() 0
end species
begin reactions
    1 1,2 1,3 MM kcat Km
    2 3 0 kd
end reactions
begin groups
    1 Ptot 3
end groups
"""


@pytest.mark.parametrize("kind", X.TRANSFORMS)
def test_a_net_rate_law_keyword_and_a_spaced_expression_keep_their_answer(kind):
    """BNG's MM kcat Km is a keyword and its arguments, linear in kcat (split
    and rescale_time scale that, mm_tqssa failed to load before), and kd's value
    holds spaces (rename substituted only its first word, ic_derived_compound)."""
    path = _DIR / "keyword.net"
    path.write_text(_KEYWORD_NET)
    v = D.check_transform(_job_of("keyword", path), kind)
    assert v.status == "pass", v


def test_a_parameter_named_like_an_exponent_is_renamed_outside_numbers_only():
    """rename's substitution read the exponent of 2e1 as a parameter named e1,
    and wrote 2mt_p1, which does not load."""
    assert X._subst("2e1*e1 + 1E0*E0 + 1.5e1", {"e1": "p", "E0": "q"}) == "2e1*p + 1E0*q + 1.5e1"
    path = _DIR / "exponent_names.net"
    path.write_text(
        "begin parameters\n    1 e1 0.5\n    2 k 2e1*e1\nend parameters\n"
        "begin species\n    1 A() 10\n    2 B() 0\nend species\n"
        "begin reactions\n    1 1 2 k\nend reactions\n"
    )
    v = D.check_transform(_job_of("exponent_names", path), "rename")
    assert v.status == "pass", v


def test_the_transforms_run_after_every_check_main_runs():
    """A model's checks share its cap (#1321): four more runs ahead of main's
    checks could tip a slow model's locale row, which the baseline holds, to
    TIMEOUT. They run after them, as every check added since does."""
    main = (
        "sens_runs",
        "sens_vs_plain",
        "ajac_vs_fdjac",
        "klu_vs_dense",
        "cc_vs_interp",
        "cc_vs_mir",
        "locale",
        "grid",
        "sparsity_covers_jacobian",
        "backend_as_asked",
    )
    heard = []
    D.check_all(
        SHAPES["equality-dwell"],
        None,
        (*main, *X.TRANSFORMS),
        on_verdict=lambda c, v: heard.append(c),
    )
    assert set(heard) == {*main, *X.TRANSFORMS}, heard
    assert max(heard.index(c) for c in main) < min(heard.index(k) for k in X.TRANSFORMS), heard


def test_an_id_bngsim_respells_is_aligned_by_its_respelling():
    """An SBML id its expressions reserve (null, In, time) is reported as
    _ant_<id>; renamed, the map back gives the id, and twenty BioModels read as
    species missing before."""
    import libsbml

    doc = libsbml.readSBMLFromFile(str(_AUTONOMOUS.path))
    doc.getModel().getSpecies("P").setId("null")
    for el in doc.getModel().getListOfAllElements():
        el.renameSIdRefs("P", "null")
    path = _DIR / "respelled.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    base = _job_of("respelled", path, 20.0)
    assert "_ant_null" in base.species_names, base.species_names
    v = D.check_transform(base, "rename")
    assert v.status == "pass", v


def test_a_backend_other_than_the_one_asked_is_a_finding(monkeypatch):
    """#1213: codegen=False was ignored on MIR builds. Here the property lies."""
    monkeypatch.setattr(bngsim.Simulator, "codegen_backend", property(lambda self: "mir"))
    v = D.check_backend(_base("triangle-maxabs"))
    assert v.status == "diff" and "codegen=False ran mir" in v.note, v


def test_a_sensitivity_run_refused_is_unsupported_not_a_pass():
    v = _verdicts("equality-dwell")["sens_runs"]
    assert v.status == "unsupported" and "SensitivityUnsupportedError" in v.note


@functools.cache
def _base(shape):
    return D.Base(SHAPES[shape])


def test_the_conditioning_gate_keeps_a_chaotic_model_from_a_false_finding(monkeypatch):
    """Lorenz over t = 25 amplifies every difference: the FD Jacobian's path is
    3.4e-4 from the base, past the 1e-5 that 1e3·rtol allows, and both are
    right, for the base moves by 0.79 at a tenfold tighter tolerance itself. Such a
    base is no reference at all, and nothing is judged against it; with the
    drift as the bound alone, it passes; with neither, it is a finding."""
    job = D.Job(
        "lorenz",
        _sbml(
            "lorenz",
            "s = 10; r = 28; b = 2.6666666666666665; X = 1; Y = 1; Z = 1;"
            " X' = s*(Y - X); Y' = X*(r - Z) - Y; Z' = X*Y - b*Z;",
        ),
        0.0,
        25.0,
        21,
        1e-8,
        1e-10,
    )
    gated = D.check_ajac(D.Base(job))
    assert gated.status == "skip" and "no reference" in gated.note, gated
    monkeypatch.setattr(D, "UNJUDGEABLE_DRIFT", float("inf"))
    bounded = D.check_ajac(D.Base(job))
    assert bounded.status == "pass", bounded
    monkeypatch.setattr(D, "DRIFT_FACTOR", 0.0)
    ungated = D.check_ajac(D.Base(job))
    assert ungated.status == "diff", ungated


def test_codegen_true_that_runs_the_interpreter_is_a_finding(monkeypatch):
    """#1213's rule, its other half: codegen=True builds compiled code or raises,
    so a run on the interpreter is a finding, not a machine without a compiler."""
    monkeypatch.setattr(bngsim.Simulator, "codegen_backend", property(lambda self: "exprtk"))
    base = D.Base(SHAPES["triangle-maxabs"])
    v = D.check_backend(base)
    assert v.status == "diff" and "codegen=True ran exprtk" in v.note, v
    # and cc_vs_interp does not pass the interpreter against itself
    assert D.check_cc_vs_interp(base).status == "skip"


@pytest.mark.parametrize(
    "text",
    [
        # Windows' compiler lookup, a missing cc, and macOS's shim without the CLT
        "No C compiler found. Install Visual Studio Build Tools (cl.exe) or MinGW (gcc).",
        "C compiler not found. Install Xcode Command Line Tools (macOS), gcc (Linux), ...",
        "Codegen compilation failed:\nxcrun: error: invalid active developer path "
        "(/Library/Developer/CommandLineTools), missing xcrun at: ...",
    ],
)
def test_a_machine_without_a_compiler_skips_the_compiled_paths(monkeypatch, text):
    base = D.Base(SHAPES["triangle-maxabs"])
    real = D.run

    def no_cc(job, **kw):
        if kw.get("codegen") is True:
            raise RuntimeError(
                f"codegen=True requested, but the codegen build failed (RuntimeError: {text})."
            )
        return real(job, **kw)

    monkeypatch.setattr(D, "run", no_cc)
    for check in (D.check_cc_vs_interp, D.check_backend):
        v = check(base)
        assert v.status == "skip" and "no C compiler" in v.note, v


def _chain(n=60) -> D.Job:
    """A60 -> ... in a line: 60 species and a sparse Jacobian, which the
    automatic rule sends to KLU."""
    params = "\n".join(f" {i + 1} k{i} {0.5 + 0.01 * i}" for i in range(n))
    species = "\n".join(f" {i + 1} A{i}() {10.0 if i == 0 else 0.0}" for i in range(n))
    reactions = "\n".join(f" {i + 1} {i + 1} {i + 2} k{i}" for i in range(n - 1))
    return D.Job("chain", _net("chain", params, species, "", reactions, f" 1 Aend {n}"))


@pytest.mark.skipif(not bngsim.HAS_KLU, reason="KLU not compiled")
def test_klu_is_compared_with_the_dense_solver_where_the_auto_rule_picks_klu(monkeypatch):
    """The base of a sparse model of 50 species or more runs on KLU: a reference
    left to the automatic rule is KLU against KLU."""
    base = D.Base(_chain())
    solvers = []
    real = D.run

    def spy(job, **kw):
        result, sim = real(job, **kw)
        solvers.append(result.solver_stats["linear_solver"])
        return result, sim

    monkeypatch.setattr(D, "run", spy)
    v = D.check_klu(base)
    assert v.status == "pass", v
    assert len(solvers) == 2 and solvers.count(1) == 1, solvers  # 1 is KLU


def test_a_tighter_run_that_fails_leaves_the_rules_that_compare_nothing(monkeypatch):
    """A base whose tenfold tighter run does not complete is no reference, and
    its comparisons are SKIP; the model still ran, and the rules that compare
    no answers (sparsity, backend) still apply."""

    def fails(job, x):
        raise bngsim.SimulationError("too much work")

    monkeypatch.setattr(D, "own_drift", fails)
    v = D.check_all(SHAPES["triangle-maxabs"], None, ("ajac_vs_fdjac", "sparsity_covers_jacobian"))
    assert v["ajac_vs_fdjac"].status == "skip" and "did not complete" in v["ajac_vs_fdjac"].note
    assert v["sparsity_covers_jacobian"].status == "pass", v


def test_a_nanomolar_state_is_stepped_on_its_own_scale():
    """A step floor of 1e-3 absolute moved a state of 2e-9 by half its size, and
    a saturating law's two steps then disagreed past 10%: the missing slot read
    as unresolvable."""

    class Nano(_FakeModel):
        @staticmethod
        def rhs(y, t=0.0):
            a, b = y
            return np.array([-a * b / (1e-10 + b), -a])

    y = [np.array([1e-9, 2e-9])]
    assert D.sparsity_misses(Nano(_csc([[1, 1], [1, 0]])), y) == []
    assert D.sparsity_misses(Nano(_csc([[1, 0], [1, 0]])), y) == [(0, 1)]


def test_a_trajectory_wholly_under_the_floor_is_not_a_pass():
    """A BioModel in molar units peaks under 1e3·atol: every path inside the
    floor read PASS, a path of all zeros included. It is SKIP, said why; a path
    that clears the floor is still judged."""
    base = D.Base(SHAPES["triangle-maxabs"])
    tiny = base.against(base.x * 1e-12)  # a peak of ~1e-11 under a floor of 1e-7
    v = tiny.compare(np.zeros_like(base.x))
    assert v.status == "skip" and "under the absolute floor" in v.note, v
    assert tiny.compare(tiny.x + 1.0).status == "diff"


def test_a_state_the_rule_cannot_reach_is_left_out_not_an_exception(monkeypatch):
    """The sparsity rule reads states after a one-interval run to the middle and
    the end of the window. Where that run fails and the base's grid does not
    (MODEL1112050001), the pattern is asked at the states that were reached."""
    base = D.Base(SHAPES["triangle-maxabs"])
    real = bngsim.Simulator.run

    def fails_on_one_interval(self, *a, **kw):
        if kw.get("n_points") == 2:
            raise bngsim.SimulationError("CVODE integration failed (CV_CONV_FAILURE)")
        return real(self, *a, **kw)

    monkeypatch.setattr(bngsim.Simulator, "run", fails_on_one_interval)
    v = D.check_sparsity(base)
    assert v.status == "pass" and "at 1 states" in v.note, v


def test_a_run_past_the_models_deadline_does_not_start(monkeypatch):
    import time

    monkeypatch.setattr(D, "_DEADLINE", time.monotonic() - 1.0)
    with pytest.raises(bngsim.SimulationTimeout):
        D.run(SHAPES["triangle-maxabs"])


def test_a_cap_spent_after_the_base_times_out_each_check_alone(monkeypatch):
    """The pool kills a model's worker at its cap, and every row with it: on
    the nightly MODEL1006230049, 0053 and 0077 take 716-751 s of 900, so a check
    added to every model turned all their rows TIMEOUT. Past the deadline each
    check is its own TIMEOUT, said why, and none is an exception or a pass."""
    import time

    real = D.Base

    def base_then_spent(job):
        b = real(job)
        D._DEADLINE = time.monotonic() - 1.0
        return b

    monkeypatch.setattr(D, "Base", base_then_spent)
    v = D.check_all(SHAPES["triangle-maxabs"], None, CHECKS, deadline=time.monotonic() + 1e3)
    assert {x.status for x in v.values()} == {"timeout"}, v
    assert v["grid"].status == "timeout", v["grid"]
    assert "not finished within the model's cap" in v["grid"].note
    assert v["sens_runs"].status == "timeout", v["sens_runs"]
    assert D._DEADLINE is None


def test_no_check_starts_past_the_deadline(monkeypatch):
    """backend_as_asked builds and compiles Simulators without running one, so no
    run's timeout bounds it: on MODEL1006230049 it started past a 180 s
    deadline and took the worker to 203 s, past a 200 s cap."""
    import time

    real = D.Base
    started = []

    def base_then_spent(job):
        b = real(job)
        D._DEADLINE = time.monotonic() - 1.0
        return b

    monkeypatch.setattr(D, "Base", base_then_spent)
    monkeypatch.setattr(D, "check_backend", lambda base: started.append(1) or D.Verdict("pass"))
    v = D.check_all(SHAPES["triangle-maxabs"], None, CHECKS, deadline=time.monotonic() + 1e3)
    assert not started and v["backend_as_asked"].status == "timeout", v["backend_as_asked"]


def test_a_base_past_the_deadline_is_a_timeout_on_every_row():
    import time

    v = D.check_all(SHAPES["triangle-maxabs"], None, CHECKS, deadline=time.monotonic() - 1.0)
    assert {x.status for x in v.values()} == {"timeout"}, v


def test_no_deadline_bounds_nothing():
    assert D._DEADLINE is None
    v = D.check_all(SHAPES["triangle-maxabs"], None, ("grid",))
    assert v["grid"].status == "pass", v


def test_the_runner_hands_each_model_its_cap(tmp_path):
    """The worker's deadline is the model's cap, short of it: with no time left
    every row is a TIMEOUT of its own, not a model the pool killed."""
    import queue

    import differential_run as RUN

    job = SHAPES["triangle-maxabs"]
    spec = {
        "key": job.model_id,
        "job": job.as_dict(),
        "checks": ["grid"],
        "locale": None,
        "max_species": 200,
        "timeout": 1e-6,
    }
    q = queue.Queue()
    RUN._worker(spec, q)
    res = q.get_nowait()
    assert res["status"] == "done", res
    assert res["rows"]["grid"]["status"] == "timeout", res


def test_a_model_killed_at_the_cap_keeps_the_rows_it_finished(tmp_path):
    """A check compiling C past the deadline has no run to bound, and the pool
    kills its worker at the cap: the verdicts it reached before stand, and only
    the rest are TIMEOUT."""
    import differential_run as RUN

    job = SHAPES["triangle-maxabs"]
    spec = {
        "key": job.model_id,
        "job": job.as_dict(),
        "checks": ["grid", "backend_as_asked"],
        "timeout": 900.0,
        "partial": str(tmp_path / "0.json"),
    }
    RUN._keep(spec, "grid", D.Verdict("pass", 0.0, "agrees"))
    rows = {r.method: r for r in RUN.to_results(spec, {"status": "timeout"}, {})}
    assert rows["grid"].outcome == "PASS", rows["grid"]
    assert rows["backend_as_asked"].outcome == "TIMEOUT"
    assert rows["backend_as_asked"].exception == "cap of 900 s"


@pytest.mark.parametrize(
    ("r", "note"),
    [
        ({"status": "dead", "exitcode": -11}, "the worker died (exit code -11)"),
        ({"status": "exception", "why": "ValueError: boom"}, "ValueError: boom"),
    ],
    ids=["died", "raised"],
)
def test_a_model_that_died_or_raised_keeps_the_rows_it_finished(tmp_path, r, note):
    """A worker that segfaults, or raises past check_all, has finished what it
    kept as surely as one killed at the cap. The rest say what happened to the
    worker: one that died did not reach the cap."""
    import differential_run as RUN

    job = SHAPES["triangle-maxabs"]
    spec = {
        "key": job.model_id,
        "job": job.as_dict(),
        "checks": ["grid", "backend_as_asked"],
        "timeout": 900.0,
        "partial": str(tmp_path / "0.json"),
    }
    RUN._keep(spec, "grid", D.Verdict("pass", 0.0, "agrees"))
    rows = {x.method: x for x in RUN.to_results(spec, r, {})}
    assert rows["grid"].outcome == "PASS", rows["grid"]
    assert rows["backend_as_asked"].outcome == "EXCEPTION"
    assert rows["backend_as_asked"].exception == note


def _locale_runs_out(monkeypatch, left):
    """``check_locale`` on a child that never finishes, with ``left`` seconds of
    the model's cap (None for no cap): the timeout the child was given, and the
    verdict."""
    import subprocess
    import time

    base = D.Base(SHAPES["triangle-maxabs"])
    given = []

    def never_finishes(cmd, timeout=None, **kw):
        given.append(timeout)
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(D.subprocess, "run", never_finishes)
    monkeypatch.setattr(D, "_DEADLINE", None if left is None else time.monotonic() + left)
    return given, D.check_locale(base, "de_DE.UTF-8", timeout=600.0)


def test_a_locale_run_past_the_models_cap_is_a_timeout(monkeypatch):
    given, v = _locale_runs_out(monkeypatch, left=30.0)
    assert given[0] <= 30.0, given
    assert v.status == "timeout" and "the model's cap" in v.note, v


@pytest.mark.parametrize("left", [None, 1e4], ids=["no-cap", "cap-to-spare"])
def test_a_locale_run_past_its_own_limit_is_not_the_caps(monkeypatch, left):
    """The run under C finished; the same run under a comma-decimal locale
    did not, in 600 s. That stays an EXCEPTION, as before the cap, and not a
    TIMEOUT the baseline reads as a changed state: a locale that stalls a run is
    the check's finding (#1235's class)."""
    given, v = _locale_runs_out(monkeypatch, left=left)
    assert given == [600.0], given
    assert v.status == "exception" and "TimeoutExpired" in v.note, v


def test_each_verdict_is_heard_as_it_is_reached():
    heard = {}
    v = D.check_all(SHAPES["triangle-maxabs"], None, CHECKS, on_verdict=heard.__setitem__)
    assert heard.keys() == v.keys()
    assert all(heard[c] is v[c] for c in v)


@pytest.mark.parametrize(
    "error",
    [
        # codegen=True's refusal: the compile's timeout in its text, no cause
        RuntimeError(
            "codegen=True requested, but the codegen build failed (RuntimeError: Codegen "
            "compilation timed out after 600 s (4.2 MB C source at -O1). The RHS, ..."
        ),
        # the sensitivity build's: chained to the compile's TimeoutExpired
        "chained",
    ],
    ids=["in-text", "chained"],
)
def test_a_compile_past_bngsims_limit_is_a_timeout_not_a_crash(error):
    """BIOMD0000000385 compiles 4.2 MB of C, past bngsim's 600 s limit: given
    the time to reach that compile, it read TIMEOUT -> EXCEPTION, a "new crash"
    the nightly alerts on. A timeout, whatever wraps it."""
    import subprocess

    if error == "chained":
        try:
            try:
                raise subprocess.TimeoutExpired(["cc"], 600)
            except subprocess.TimeoutExpired as inner:
                raise RuntimeError("Codegen compilation timed out after 600 s") from inner
        except RuntimeError as compile_error:
            try:
                raise RuntimeError("Failed to build the analytical sensitivity RHS") from (
                    compile_error
                )
            except RuntimeError as outer:
                error = outer
    v = D._verdict_of(error)
    assert v.status == "timeout", v
    assert D._verdict_of(RuntimeError("Codegen compilation failed: syntax")).status == "exception"


# ── The SSA arm (#1310): compiled propensities against interpreted ones ──

_DIMER = """begin parameters
    1 k 0.01
    2 kd 0.5
end parameters
begin species
    1 A() 100
    2 B() 0
end species
begin reactions
    1 1,1 2 k
    2 2 1,1 kd
end reactions
"""


def _ssa_job(n_rep=10):
    path = _DIR / "dimer_ssa.net"
    path.write_text(_DIMER)
    return D.Job("dimer", path, 0.0, 5.0, 11, method="ssa", n_rep=n_rep, seed=1)


def _compiled_here():
    r = bngsim.Simulator(D.load(_ssa_job().path), method="ssa").run(
        t_span=(0, 1), n_points=2, seed=1
    )
    return r.ssa_diagnostics["propensity_backend"] != "interpreted"


def test_an_ssa_job_runs_the_ssa_checks_and_an_ode_job_does_not():
    assert set(D.checks_for(_ssa_job())) == set(D.SSA_CHECKS)
    assert not set(D.checks_for(SHAPES["equality-dwell"])) & set(D.SSA_CHECKS)


def test_compiled_and_interpreted_mass_action_agree_draw_for_draw():
    if not _compiled_here():
        pytest.skip("no C compiler: the default SSA runs interpreted too")
    v = D.check_all(_ssa_job(), None)
    assert v["ssa_cc_vs_interp"].status == "pass" and "draw for draw" in v["ssa_cc_vs_interp"].note
    assert v["ssa_backend_as_asked"].status == "pass", v["ssa_backend_as_asked"]


def _perturbed_compiled(monkeypatch, factor):
    """One propensity perturbed, on the compiled side only: its rate constant
    k scaled by ``factor`` in the models the compiled runs load."""
    real_runs, real_load = D._ssa_runs, D.load
    on = []

    def load(path):
        model = real_load(path)
        if on:
            model.set_param("k", factor * float(model.get_param("k")))
        return model

    def runs(job, compiled, n=None):
        on[:] = [True] if compiled else []
        try:
            return real_runs(job, compiled, n)
        finally:
            on[:] = []

    monkeypatch.setattr(D, "load", load)
    monkeypatch.setattr(D, "_ssa_runs", runs)


def test_one_perturbed_propensity_is_a_finding(monkeypatch):
    """The issue's negative control: mass-action laws alone, so one draw off
    is a DIFF. 1% off one rate constant moves draws in every seed here (a
    change of 1e-9 moves none: it shifts each event time by as little, which
    crosses no sample)."""
    if not _compiled_here():
        pytest.skip("no C compiler: the default SSA runs interpreted too")
    _perturbed_compiled(monkeypatch, 1.01)
    v = D.check_all(_ssa_job(), None)["ssa_cc_vs_interp"]
    assert v.status == "diff" and "mass-action" in v.note, v


def test_functional_laws_are_judged_in_distribution(monkeypatch):
    """Where rounding in a functional law may change a selection, the seeds
    may differ and the ensembles must not: the same ensemble at other seeds
    passes, one with a propensity doubled does not."""
    if not _compiled_here():
        pytest.skip("no C compiler: the default SSA runs interpreted too")
    monkeypatch.setattr(D, "_all_elementary", lambda job: False)
    real = D._ssa_runs

    def other_seeds(job, compiled, n=None):
        if compiled:
            job = D.Job(**{**job.__dict__, "seed": job.seed + 1000})
        return real(job, compiled, n)

    monkeypatch.setattr(D, "_ssa_runs", other_seeds)
    same = D.check_all(_ssa_job(n_rep=30), None)["ssa_cc_vs_interp"]
    assert same.status == "pass" and "in distribution" in same.note, same
    monkeypatch.setattr(D, "_ssa_runs", real)
    _perturbed_compiled(monkeypatch, 3.0)
    moved = D.check_all(_ssa_job(n_rep=30), None)["ssa_cc_vs_interp"]
    assert moved.status == "diff" and "in distribution" in moved.note, moved


def test_codegen_false_that_runs_compiled_propensities_is_a_finding(monkeypatch):
    real = D._ssa_runs

    def lies(job, compiled, n=None):
        out, backends = real(job, compiled, n)
        return out, ({"cc"} if not compiled else backends)

    monkeypatch.setattr(D, "_ssa_runs", lies)
    v = D.check_all(_ssa_job(), None)["ssa_backend_as_asked"]
    assert v.status == "diff" and "codegen=False ran" in v.note, v


def test_a_model_without_a_propensity_library_runs_no_replicate(monkeypatch):
    """Most of the corpus is not eligible for compiled propensities, which the
    Simulator decides before any run: there is nothing to tell apart, and the
    corpus's slowest take one replicate past the nightly's cap."""
    import bngsim._codegen

    asked = []
    monkeypatch.setattr(bngsim._codegen, "prepare_ssa_propensity_lib", lambda model: None)
    monkeypatch.setattr(D, "_ssa_runs", lambda *a, **kw: asked.append(a) or ([], set()))
    v = D.check_all(_ssa_job(n_rep=10), None)
    assert asked == [], asked
    assert {c: x.status for c, x in v.items()} == dict.fromkeys(D.SSA_CHECKS, "skip"), v


def test_a_library_the_engine_declines_takes_one_replicate_each_way(monkeypatch):
    """Where the Simulator built a library and the engine still runs the
    interpreter (it declines events, functional laws and large networks),
    one replicate of each decides both checks, not an ensemble of each."""
    if not _compiled_here():
        pytest.skip("no C compiler: no propensity library is built")
    real, asked = D._ssa_runs, []

    def interpreted(job, compiled, n=None):
        asked.append((compiled, job.n_rep if n is None else n))
        out, _ = real(job, False, n)
        return out, {"interpreted"}

    monkeypatch.setattr(D, "_ssa_runs", interpreted)
    v = D.check_all(_ssa_job(n_rep=10), None)
    assert asked == [(True, 1), (False, 1)], asked
    assert {c: x.status for c, x in v.items()} == dict.fromkeys(D.SSA_CHECKS, "skip"), v


def test_an_ssa_job_with_no_replicates_is_skipped_not_judged(tmp_path):
    """Job's n_rep defaults to 0: two empty ensembles read PASS, "all 0 seeds
    identical", and no backend read DIFF, "codegen=False ran []". Nothing was
    compared, so both are SKIP, said why; the runner refuses --max-reps 0."""
    import differential_run as RUN

    v = D.check_all(_ssa_job(n_rep=0), None)
    assert {c: x.status for c, x in v.items()} == dict.fromkeys(D.SSA_CHECKS, "skip"), v
    assert all("no replicates" in x.note for x in v.values()), v
    # An empty corpus, so a runner that took --max-reps 0 returns at once.
    empty = ["--fd-sens-jobs", "", "--rr-jobs", "", "--ssa-jobs", "", "--no-locale"]
    with pytest.raises(SystemExit):
        RUN.main([*empty, "--max-reps", "0", "--out", str(tmp_path / "r.json")])


def test_the_runner_reads_rr_ssas_jobs_as_ssa_jobs(tmp_path):
    import differential_run as RUN

    (tmp_path / "m.net").write_text(_DIMER)
    manifest = tmp_path / "ssa_jobs.json"
    params = {"t_start": 0.0, "t_end": 5.0, "n_points": 1001, "seed": 7, "n_rep": 50}
    manifest.write_text(
        json.dumps({"jobs": [{"model_id": "m", "model": "m.net", "params": params}]})
    )
    (job,) = RUN.ssa_jobs(manifest, 0, 101, 20)
    assert (job.method, job.n_points, job.n_rep, job.seed) == ("ssa", 101, 20, 7)


# The NF arm (#1312): NFsim against RuleMonkey on one BNG-XML.
_NF_XML = _ROOT / "tests" / "data" / "nfsim" / "simple_system.xml"


def _nf_job(n_rep=10):
    if not _NF_XML.is_file():
        pytest.skip(f"test data not found: {_NF_XML}")
    if not bngsim._bngsim_core.HAS_RULEMONKEY:
        pytest.skip("RuleMonkey not compiled in")
    sim = {"block_same_complex_binding": True}
    return D.Job("simple_system", _NF_XML, 0.0, 10.0, 11, sim=sim, method="nf", n_rep=n_rep)


def _faster_on_rulemonkey(monkeypatch, tmp_path, factor):
    """RuleMonkey reads a copy of the XML with ``kcat`` times ``factor``; NFsim
    the original: one rate perturbed on one engine, the issue's control."""
    text = _NF_XML.read_text()
    old = '<Parameter id="kcat" type="Constant" value="0.7"/>'
    assert old in text
    moved = tmp_path / "simple_system_kcat.xml"
    moved.write_text(text.replace(old, old.replace("0.7", repr(0.7 * factor))))
    real = D._nf_runs

    def runs(job, engine, n=None, seed=None):
        if engine == "nf_exact":
            job = D.Job(**{**job.__dict__, "path": moved})
        return real(job, engine, n, seed)

    monkeypatch.setattr(D, "_nf_runs", runs)


def test_an_nf_job_runs_the_nf_check_and_no_other():
    job = D.Job("m", _NF_XML, 0.0, 1.0, 2, method="nf", n_rep=2)
    assert D.checks_for(job) == D.NF_CHECKS
    assert not set(D.checks_for(SHAPES["equality-dwell"])) & set(D.NF_CHECKS)
    assert not set(D.checks_for(_ssa_job())) & set(D.NF_CHECKS)


def test_nfsim_and_rulemonkey_agree_in_distribution():
    v = D.check_all(_nf_job(), None)["nf_reject_vs_exact"]
    assert v.status == "pass" and "replicates each" in v.note, v


def test_an_nf_job_with_no_replicates_is_skipped_not_no_base():
    """With n_rep 0 there is no ensemble to stack, which read as NFsim failing
    ("the NFsim runs: ValueError: need at least one array"): SKIP, said why."""
    v = D.check_all(_nf_job(n_rep=0), None)["nf_reject_vs_exact"]
    assert v.status == "skip" and "no replicates" in v.note, v


def test_one_rate_perturbed_on_one_engine_is_a_finding(monkeypatch, tmp_path):
    """The issue's negative control. A DIFF is escalated to 30 replicates each,
    as the stochastic suites do, and a real one stays a DIFF."""
    _faster_on_rulemonkey(monkeypatch, tmp_path, 3.0)
    seeds, real_init = [], bngsim.NfsimSession.initialize

    def initialize(self, seed=None):
        seeds.append(seed)
        return real_init(self, seed=seed)

    monkeypatch.setattr(bngsim.NfsimSession, "initialize", initialize)
    v = D.check_all(_nf_job(), None)["nf_reject_vs_exact"]
    assert v.status == "diff", v
    assert [n for n, _ in v.extra["rungs"]][:2] == [10, 30], v.extra
    # the rung's replicates are new draws, not the base's again
    assert seeds[:30] == list(range(1, 31)), seeds
    assert v.extra["worst_observable"] in ("X_p_total", "Xp_free", "X_free"), v.extra


def test_an_escalation_past_its_budget_is_not_walked(monkeypatch, tmp_path):
    """A slow model's DIFF is not walked to 100 replicates each past the
    nightly's cap: the row stays at its base rung and says so."""
    _faster_on_rulemonkey(monkeypatch, tmp_path, 3.0)
    monkeypatch.setattr(D, "NF_ESCALATION_BUDGET", 0.0)
    v = D.check_all(_nf_job(), None)["nf_reject_vs_exact"]
    assert v.status == "diff" and "aborted" in v.note, v
    assert [n for n, _ in v.extra["rungs"]] == [10], v.extra


def test_engines_reporting_different_observables_is_a_finding(monkeypatch):
    real = D._nf_runs

    def renamed(job, engine, n=None, seed=None):
        obs, names = real(job, engine, n, seed)
        return obs, (names[::-1] if engine == "nf_exact" else names)

    monkeypatch.setattr(D, "_nf_runs", renamed)
    v = D.check_all(_nf_job(), None)["nf_reject_vs_exact"]
    assert v.status == "diff" and "the observables differ" in v.note, v


def test_a_model_nfsim_does_not_run_is_no_reference(monkeypatch):
    """NFsim is the base: where it fails there is nothing to compare RuleMonkey
    against, and bng_parity's NF track owns the failure."""

    def fails(*a, **kw):
        raise bngsim.SimulationError("NFsim: traversal limit exceeded")

    monkeypatch.setattr(bngsim, "NfsimSession", fails)
    v = D.check_all(_nf_job(), None)["nf_reject_vs_exact"]
    assert v.status == "no_base" and "the NFsim runs" in v.note, v


@pytest.mark.parametrize(
    "refusal",
    [
        bngsim.ModelError("RuleMonkey does not support compartments (cBNGL)"),
        # ft_total_rate, in bng_nf's corpus
        RuntimeError("RuleMonkey cannot honor this BNG XML: RateLaw@totalrate: Rule 'RR1'"),
    ],
    ids=["compartments", "total-rate"],
)
def test_rulemonkey_refusing_a_model_is_unsupported(monkeypatch, refusal):
    def refuses(*a, **kw):
        raise refusal

    monkeypatch.setattr(bngsim, "RuleMonkeySession", refuses)
    v = D.check_all(_nf_job(), None)["nf_reject_vs_exact"]
    assert v.status == "unsupported" and str(refusal)[:20] in v.note, v


def test_the_runner_reads_bng_nfs_archive_as_nf_jobs(tmp_path):
    """Each entry carries the XML BNG2.pl wrote and the run it was recorded for;
    a multi-segment protocol is left out (one segment is not its run)."""
    import differential_run as RUN

    def entry(name, mode):
        meta = {
            "track": "nf",
            "model_id": name,
            "n_rep": 10,
            "seed_base": 3,
            "nf_bscb": True,
            "build": {"artifact_text": "<xml/>"},
            "run": {"t_start": 0.0, "t_end": 50.0, "n_steps": 100, "gml": 5000, "mode": mode},
        }
        blob = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        np.savez_compressed(tmp_path / f"{name}.npz", meta=blob)

    entry("one", "single_segment")
    entry("two", "multi_segment")
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    (job,) = RUN.nf_jobs(tmp_path, 20, xml_dir)
    assert (job.model_id, job.method, job.t_end, job.n_points, job.n_rep, job.seed) == (
        "one",
        "nf",
        50.0,
        101,
        10,
        3,
    )
    assert job.sim == {"block_same_complex_binding": True, "molecule_limit": 5000}
    assert job.path.read_text() == "<xml/>"


# ── Generated models and the shrinker (#1306) ──

import _generate as G  # noqa: E402
import differential_generate as GEN  # noqa: E402


def test_a_seed_is_its_model():
    """A finding is reproduced from its seed alone."""
    for fmt in G.FORMATS:
        a, b = G.generate(7, fmt), G.generate(7, fmt)
        assert (
            G.render_net(a) == G.render_net(b)
            if fmt == "net"
            else G.render_sbml(a) == G.render_sbml(b)
        )
    assert G.render_net(G.generate(7, "net")) != G.render_net(G.generate(8, "net"))


# Every (format, entry) that has a spelling: a .net-only entry (rint, erf...) is
# not drawn for SBML, so there is nothing of it to read there.
_SPELLED = [
    (fmt, op)
    for fmt in G.FORMATS
    for op in sorted(G.GRAMMAR)
    if G.GRAMMAR[op][1 if fmt == "net" else 2] is not None
]


@pytest.mark.parametrize(("fmt", "op"), _SPELLED)
def test_every_grammar_entry_is_one_bngsim_reads(fmt, op):
    """Each spelling, alone in a law, loads and integrates to a finite answer:
    a generator that writes what bngsim refuses finds nothing but refusals. Its
    first argument is negative (0 - S1), which is what each guard is for: an
    unguarded log or square root of it is NaN."""
    negative = G.Node("sub", [G.Node("num", value=0.0), G.Node("obs", value=0)])
    leaves = [negative, G.Node("par", value=0), G.Node("num", value=1.5), G.Node("obs", value=1)]
    law = G.Node(op, leaves[: G.GRAMMAR[op][0]])
    model = G.Model(fmt, [2.0, 1.0], [0.7, 0.4], [G.Reaction([0], [1], law)])
    # Named by position, not by op: "if<" is no file name on Windows.
    path = G.write(model, _DIR / f"grammar_{fmt}_{_SPELLED.index((fmt, op))}")
    load = bngsim.Model.from_net if fmt == "net" else bngsim.Model.from_sbml
    r = bngsim.Simulator(load(str(path)), method="ode").run(t_span=(0, 5), n_points=6)
    assert np.isfinite(np.asarray(r.species)).all()


@pytest.mark.parametrize("fmt", G.FORMATS)
def test_generated_models_load_and_stay_finite(fmt):
    """The guards hold: a law is finite on the window, so a finding is a
    disagreement and not a domain error."""
    load = bngsim.Model.from_net if fmt == "net" else bngsim.Model.from_sbml
    for seed in range(12):
        path = G.write(G.generate(seed, fmt), _DIR / f"generated_{fmt}_{seed}")
        r = bngsim.Simulator(load(str(path)), method="ode").run(t_span=(0, 10), n_points=11)
        assert np.isfinite(np.asarray(r.species)).all(), (fmt, seed)


def _ops(model):
    return [n.op for rx in model.reactions for _, n in G._nodes(rx.law)]


def test_the_shrinker_cuts_down_to_what_still_finds():
    """A predicate that holds of any model with a floor() in it: the shrinker
    keeps one, and drops every reaction and node it can."""
    seed = next(s for s in range(200) if "floor" in _ops(G.generate(s, "net")))
    model = G.generate(seed, "net")

    def holds(m):
        return "floor" in _ops(m)

    small = G.shrink(model, holds)
    assert holds(small)
    assert len(small.reactions) == 1
    assert G.size(small) <= 3 < G.size(model), (G.size(small), G.size(model))


def test_the_shrinker_replaces_a_named_leaf_by_a_number():
    """A leaf that names a species, a parameter or the clock, replaced by a
    number, is no fewer nodes, and the shrinker refused every such cut: a
    reproducer kept its observables and its time() for nothing."""
    law = G.Node("floor", [G.Node("obs", value=0)])
    model = G.Model("net", [2.0, 1.0], [0.7, 0.4], [G.Reaction([0], [1], law)])
    small = G.shrink(model, lambda m: "floor" in _ops(m))
    assert _ops(small) == ["floor", "num"], _ops(small)


def _spec(tmp_path, seed, checks=("cc_vs_interp",)):
    return {
        "seed": seed,
        "format": "net",
        "checks": list(checks),
        "depth": 3,
        "t_end": 10.0,
        "n_points": 11,
        "rtol": 1e-8,
        "atol": 1e-10,
        "max_steps": GEN.DEFAULT_MAX_STEPS,
        "shrink_budget": GEN.DEFAULT_SHRINK_BUDGET,
        "repro_dir": str(tmp_path),
    }


def test_a_planted_defect_is_found_and_shrunk_to_a_reproducer(tmp_path, monkeypatch):
    """The runner end to end: cc_vs_interp made to read DIFF on any model whose
    text has a rint() in it, as #1212's spelling bug read DIFF on one with an
    implicit product. The seed's finding is shrunk to a model that still has
    the rint() and little else, written out where the report says."""
    seed = next(s for s in range(200) if "rint" in _ops(G.generate(s, "net")))

    def planted(base):
        if "rint(" in Path(base.job.path).read_text():
            return D.Verdict("diff", 1.0, "planted: rint() spelled differently")
        return D.Verdict("pass", 0.0, "")

    monkeypatch.setattr(D, "check_cc_vs_interp", planted)
    res = GEN.investigate(_spec(tmp_path, seed))
    assert res["status"] == "done", res
    (finding,) = res["findings"]
    assert finding["check"] == "cc_vs_interp" and finding["status"] == "diff"
    text = Path(finding["reproducer"]).read_text()
    assert "rint(" in text
    assert finding["shrunk_size"] < finding["size"]
    assert text.count("f1()") == 1 and "f2()" not in text  # one reaction left


def test_no_finding_is_reported_where_every_check_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "check_cc_vs_interp", lambda base: D.Verdict("pass", 0.0, ""))
    res = GEN.investigate(_spec(tmp_path, 3))
    assert res["status"] == "done" and res["findings"] == [], res


def _plant_rint(monkeypatch):
    """cc_vs_interp reads DIFF on any model whose text has a rint() in it."""

    def planted(base):
        if "rint(" in Path(base.job.path).read_text():
            return D.Verdict("diff", 1.0, "planted: rint() spelled differently")
        return D.Verdict("pass", 0.0, "")

    monkeypatch.setattr(D, "check_cc_vs_interp", planted)


def test_a_finding_is_written_before_it_is_shrunk(tmp_path, monkeypatch):
    """The seed's cap covers the shrink, up to its budget of check runs: a
    seed killed mid-shrink left no reproducer at all. The model as found is
    written first, its note saying it is not shrunk."""
    import json

    seed = next(s for s in range(200) if "rint" in _ops(G.generate(s, "net")))
    _plant_rint(monkeypatch)

    def killed(*a, **kw):
        raise RuntimeError("killed mid-shrink (planted)")

    monkeypatch.setattr(G, "shrink", killed)
    res = GEN.investigate(_spec(tmp_path, seed))
    assert res["status"] == "error", res
    note = json.loads((tmp_path / f"net_{seed}_cc_vs_interp.json").read_text())
    assert note["shrunk_size"] is None and note["check"] == "cc_vs_interp", note
    assert "rint(" in Path(note["reproducer"]).read_text()


def test_a_worker_that_dies_is_dead_not_a_timeout(monkeypatch):
    """A worker gone without a result (a segfault in bngsim looks so) is a
    crash, the finding a generator is for; it read as a timeout, at once."""
    # print takes (spec, q), puts nothing on the queue, and returns: exit 0.
    monkeypatch.setattr(GEN, "_worker", print)
    (res,) = GEN.run([{"seed": 0, "format": "net"}], workers=1, timeout=60)
    assert res["status"] == "dead" and res["exitcode"] == 0, res


# ── State leaks (#1307): a model mutated between runs reads DIFF ──

# equality-dwell is a .net whose parameters all move it; event-at-a-sample an
# SBML model with an event, which restart crosses leg by leg.
LEAKED_ON = ("equality-dwell", "event-at-a-sample")


def _double_first_knob(model):
    knob = D.sens_params(model)[0]
    model.set_param(knob, 2.0 * float(model.get_param(knob)))


@pytest.mark.parametrize("shape", LEAKED_ON)
def test_a_model_changed_between_the_legs_of_a_restart_is_a_finding(monkeypatch, shape):
    real = bngsim.Simulator.run_until
    legs = []

    def leaky(self, t, **kw):
        if legs:
            _double_first_knob(self._model)
        legs.append(t)
        return real(self, t, **kw)

    monkeypatch.setattr(bngsim.Simulator, "run_until", leaky)
    v = D.check_restart(_base(shape))
    assert v.status == "diff" and "run_until" in v.note, v


def test_a_fresh_model_given_a_state_it_does_not_hold_is_a_finding(monkeypatch):
    """The second restart: a fresh model from the middle's saved state, which
    a model without events takes (equality-dwell has none)."""
    real = bngsim.Simulator.set_state

    def leaky(self, state, **kw):
        real(self, state, **kw)
        _double_first_knob(self._model)

    monkeypatch.setattr(bngsim.Simulator, "set_state", leaky)
    v = D.check_restart(_base("equality-dwell"))
    assert v.status == "diff" and "fresh model" in v.note, v


@pytest.mark.parametrize("shape", LEAKED_ON)
def test_a_restore_that_leaves_a_parameter_changed_is_a_finding(monkeypatch, shape):
    """Reuse rests on snapshot()/restore(): one that leaves the first run's
    mark reads as the leak it is."""
    real = bngsim.Simulator.restore

    def leaky(self, snapshot=None):
        real(self, snapshot)
        _double_first_knob(self._model)

    monkeypatch.setattr(bngsim.Simulator, "restore", leaky)
    v = D.check_reuse(_base(shape))
    assert v.status == "diff", v


def _threaded_point_1_doubled(monkeypatch):
    """Point 1 of a threaded batch run at twice its parameters."""
    real_batch, real_one = bngsim.Simulator.run_batch, bngsim.Simulator._run_single_batch
    threaded = []

    def batch(self, *a, num_processors=None, **kw):
        threaded.append(bool(num_processors and num_processors > 1))
        try:
            return real_batch(self, *a, num_processors=num_processors, **kw)
        finally:
            threaded.pop()

    def one(self, i, p, *a, **kw):
        if threaded and threaded[-1] and i == 1:
            p = {k: 2.0 * v for k, v in p.items()}
        return real_one(self, i, p, *a, **kw)

    monkeypatch.setattr(bngsim.Simulator, "run_batch", batch)
    monkeypatch.setattr(bngsim.Simulator, "_run_single_batch", one)


@pytest.mark.parametrize("shape", LEAKED_ON)
def test_a_batch_point_run_on_another_parameter_is_a_finding(monkeypatch, shape):
    """One threaded point run on a parameter other than its own, as a clone
    that shared its parent's state would be."""
    _threaded_point_1_doubled(monkeypatch)
    v = D.check_batch(_base(shape))
    assert v.status == "diff" and "point 1" in v.note, v


def test_a_skip_against_the_base_does_not_hide_a_threaded_points_diff(monkeypatch):
    """Point 0 is judged against the base, and reads SKIP where the base moves
    past UNJUDGEABLE_DRIFT at a tighter tolerance; a threaded point against
    the same point run alone needs no base, and its DIFF is the row's."""
    monkeypatch.setattr(D, "own_drift", lambda job, x: 2.0 * D.UNJUDGEABLE_DRIFT)
    base = D.Base(SHAPES["equality-dwell"])
    assert base.compare(base.x).status == "skip"
    _threaded_point_1_doubled(monkeypatch)
    v = D.check_batch(base)
    assert v.status == "diff" and "point 1" in v.note, v


def _failing_point(monkeypatch, only_threaded):
    """Point 1 of the batch raises, on threads only or always."""
    real_batch, real_one = bngsim.Simulator.run_batch, bngsim.Simulator._run_single_batch
    threaded = []

    def batch(self, *a, num_processors=None, **kw):
        threaded.append(bool(num_processors and num_processors > 1))
        try:
            return real_batch(self, *a, num_processors=num_processors, **kw)
        finally:
            threaded.pop()

    def one(self, i, p, *a, **kw):
        if i == 1 and (threaded[-1] or not only_threaded):
            raise bngsim.SimulationError(f"Batch simulation {i} failed: CVODE (planted)")
        return real_one(self, i, p, *a, **kw)

    monkeypatch.setattr(bngsim.Simulator, "run_batch", batch)
    monkeypatch.setattr(bngsim.Simulator, "_run_single_batch", one)


def test_a_sweep_point_that_does_not_run_alone_is_not_the_batchs_to_judge(monkeypatch):
    """BIOMD0000000763's: a point 10% off its parameter fails to integrate one
    at a time as on threads, which is the model, not a leak."""
    _failing_point(monkeypatch, only_threaded=False)
    v = D.check_batch(_base("equality-dwell"))
    assert v.status == "skip" and "one at a time" in v.note, v


def test_a_point_that_fails_only_on_threads_is_a_finding(monkeypatch):
    _failing_point(monkeypatch, only_threaded=True)
    v = D.check_batch(_base("equality-dwell"))
    assert v.status == "exception", v


# ── Attribution and branch stability (#1309) ──


def _verdict(ref, path):
    return D.Verdict("diff", 1.0, "", ref_x=np.asarray(ref, float), path_x=np.asarray(path, float))


def test_each_arm_names_the_side_it_takes():
    """An arm on the reference's side, one on the path's, one between: a
    split, which marks the model contested."""
    base = _base("derived-log10")
    ref = base.x
    path = ref * 1.5
    v = _verdict(ref, path)
    a = D.attribute(
        base, v, {"tighter": path.copy(), "dense": ref.copy(), "roadrunner": ref * 1.25}
    )
    assert a["verdict"] == "split"
    assert a["arms"]["tighter"]["side"] == "path"
    assert a["arms"]["dense"]["side"] == "reference"
    assert a["arms"]["roadrunner"]["side"] == "neither"
    assert D.contested({"klu_vs_dense": {"status": "diff", "extra": {"attribution": a}}})


def test_arms_that_agree_attribute_the_diff_to_one_side():
    base = _base("derived-log10")
    v = _verdict(base.x, base.x * 1.5)
    a = D.attribute(base, v, {"tighter": base.x.copy(), "dense": base.x.copy()})
    assert a["verdict"] == "reference"
    assert not D.contested({"klu_vs_dense": {"status": "diff", "extra": {"attribution": a}}})


def test_an_arm_takes_a_side_within_the_bound_and_none_past_it():
    """Arms half the bound off each side take that side; twice it off, neither.
    Arms that copy a side exactly pass under any bound at all, a far too tight
    one included, and #1304's dense arm reads neither on some grids anyway."""
    base = _base("derived-log10")
    ref, path = base.x, base.x * 1.5
    v = _verdict(ref, path)
    near, far = 1.0 + 0.5 * base.limit, 1.0 + 2.0 * base.limit
    a = D.attribute(base, v, {"tighter": path * near, "dense": ref * near})
    assert a["verdict"] == "split", a
    assert a["arms"]["tighter"]["side"] == "path" and a["arms"]["dense"]["side"] == "reference", a
    a = D.attribute(base, v, {"tighter": path * far, "dense": ref * far})
    assert a["verdict"] == "unattributed", a
    assert {s["side"] for s in a["arms"].values()} == {"neither"}, a


@pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled")
def test_klu_and_dense_are_compared_on_one_jacobian(monkeypatch):
    """``klu_vs_dense``'s dense reference is asked for the Jacobian its KLU
    path ran on: what differs is the linear solver and how each builds that
    Jacobian (KLU's FD one is colored over the sparsity), not which is asked."""
    asked = []
    real = D.run

    def spy(job, **kw):
        asked.append(kw)
        return real(job, **kw)

    monkeypatch.setattr(D, "run", spy)
    v = D.check_klu(_base("derived-log10"))
    assert v.status == "pass", v
    sparse = [kw["jacobian"] for kw in asked if kw.get("force_sparse_linear_solver")]
    dense = [kw["jacobian"] for kw in asked if kw.get("force_dense_linear_solver")]
    assert sparse and dense == sparse[-1:], asked


def test_a_diff_is_attributed_by_real_arms(monkeypatch):
    """check_all attributes every DIFF: here a planted cc_vs_interp that moves
    the path by 50%, which the real arms (the base at a hundredfold tighter
    tolerance, the other solver) all put on the reference's side."""
    base = _base("derived-log10")

    def planted(b):
        return b.compare(b.x * 1.5, "planted: ")

    monkeypatch.setattr(D, "check_cc_vs_interp", planted)
    v = D.check_all(SHAPES["derived-log10"], None, ("cc_vs_interp",))["cc_vs_interp"]
    assert v.status == "diff"
    a = v.extra["attribution"]
    assert a["verdict"] == "reference", a
    assert {"tighter"} <= set(a["arms"]), a
    assert base.x.shape == v.path_x.shape


def test_an_answer_that_jumps_with_the_tolerance_is_contested(monkeypatch):
    """A branch that flips at rtol x2: check_stable reads DIFF, and the model
    is contested."""
    base = _base("equality-dwell")
    real = D.run

    def flips(job, **kw):
        result, sim = real(job, **kw)
        if job.rtol > base.job.rtol * 1.5:
            result = type(
                "R",
                (),
                {
                    "species": np.asarray(result.species) * 1.5,
                    "species_names": result.species_names,
                },
            )()
        return result, sim

    monkeypatch.setattr(D, "run", flips)
    v = D.check_stable(base)
    assert v.status == "diff" and "rtol x2" in v.note, v
    assert D.contested({"stable": {"status": "diff", "extra": v.extra}})


def test_a_base_with_no_reference_is_not_stable_by_default(monkeypatch):
    """A base whose tenfold tighter run fails has no bound (base.limit is
    inf): the jitters are SKIP, said why, as every comparison against it is,
    and not the PASS an answer doubled at rtol x2 also read."""

    def own_drift(job, x):
        raise bngsim.SimulationError("CVODE integration failed (planted)")

    monkeypatch.setattr(D, "own_drift", own_drift)
    base = D.Base(SHAPES["equality-dwell"])
    real = D.run

    def doubled(job, **kw):
        result, sim = real(job, **kw)
        if job.rtol > base.job.rtol * 1.5:
            result = types.SimpleNamespace(
                species=np.asarray(result.species) * 2.0, species_names=result.species_names
            )
        return result, sim

    monkeypatch.setattr(D, "run", doubled)
    v = D.check_stable(base)
    assert v.status == "skip" and "no reference" in v.note, v


def test_an_answer_that_jumps_with_the_grid_is_contested(monkeypatch):
    base = _base("equality-dwell")
    real = D.run

    def flips(job, **kw):
        result, sim = real(job, **kw)
        if job.n_points == base.job.n_points + 1:
            x = np.asarray(result.species).copy()
            x[-1] *= 1.5
            result = type("R", (), {"species": x, "species_names": result.species_names})()
        return result, sim

    monkeypatch.setattr(D, "run", flips)
    v = D.check_stable(base)
    assert v.status == "diff" and "n_points+1" in v.note, v


def _zero_start(growth: str) -> D.Job:
    """S at 0 beside a decaying A, with S' = +k·S (``growth`` "1,1", S -> 2S)
    or -k·S ("0"): S = 0 is the answer either way, and an unstable one in the
    first, as BIOMD0000000908's sunitinib is."""
    path = _net(
        f"zero-start-{'grows' if growth == '1,1' else 'decays'}",
        " 1 k 4\n 2 a 0.5",
        " 1 S() 0\n 2 A() 10",
        "",
        f" 1 1 {growth} k\n 2 2 0 a",
        " 1 Sobs 1",
    )
    return D.Job(path.stem, path, 0.0, 10.0, 21)


def test_the_nudge_moves_a_held_start_by_its_share_of_a_step():
    """S, held at 0, moves by its share of atol; A, decaying from 10, not at all."""
    job = _zero_start("0")
    x0 = np.asarray(D.run(job, nudge=D.START_NUDGE)[0].species)[0]
    np.testing.assert_allclose(x0, [D.START_NUDGE * job.atol, 10.0], rtol=1e-12)


def test_only_a_species_the_equations_hold_still_is_nudged():
    """S' = 4·S holds S at 0, and only S is nudged. Not D, held too but read by
    a trigger at its threshold (BIOMD0000000285's death events); not A, which
    nothing moves (265's switches); not B, already moving."""
    model = bngsim.Model.from_antimony_string(
        "species S = 0, D = 0, A = 0, B = 1; S' = 4*S; D' = A; A' = 0; B' = -B\n"
        "E: at (D > 0): S = 0"
    )
    assert model.species_names == ["S", "D", "A", "B"]
    assert D.held_at_start(model, 0.0).tolist() == [True, False, False, False]


def test_an_unstable_zero_the_rounding_keeps_is_contested():
    """S = 0 holds exactly in both runs as written. From a start the solver
    cannot tell from it, the growing S leaves 0 by e^{4t}: 1e-13 at the first
    row, past the bound by t = 10. The decaying S stays put."""
    grows = D.check_stable(D.Base(_zero_start("1,1")))
    assert grows.status == "diff" and "a start nudged" in grows.note, grows
    assert D.contested({"stable": {"status": "diff", "extra": grows.extra}})
    decays = D.check_stable(D.Base(_zero_start("0")))
    assert decays.status == "pass", decays


def test_a_run_that_fails_from_the_nudged_start_is_contested(monkeypatch):
    base = _base("derived-log10")
    real = D.run

    def fails(job, nudge=0.0, **kw):
        if nudge:
            raise bngsim.SimulationError("CV_CONV_FAILURE (planted)")
        return real(job, **kw)

    monkeypatch.setattr(D, "run", fails)
    v = D.check_stable(base)
    assert v.status == "diff" and "from which the run fails (CV_CONV_FAILURE" in v.note, v
    assert D.contested({"stable": {"status": "diff", "extra": v.extra}})


_B1304 = {
    m: _ROOT / f"parity_checks/rr_parity/models/{m}/{m}_url.xml"
    for m in ("BIOMD0000000338", "BIOMD0000000339")
}


@pytest.mark.parametrize("model", sorted(_B1304))
@pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled")
def test_1304s_models_attribute_each_arm_to_the_side_it_takes(model):
    """#1304: XIIa = K = 0 is an exact, unstable solution of both models, and
    the KLU base keeps it. The dense LU's rounding breaks it, and which way and
    when depend on the CPU and the output grid. So the check reads a coin with
    three faces:

    * Broken upward, dense takes the activated branch, as libRoadRunner does
      (338's Fg 0.73 against KLU's 311; 339 1.8e-2 of the peak apart), and
      the dense arm lands with the reference: the arms split, and the
      model is contested. Both models at this window on arm64; 339 on x86-64.
    * Broken upward, but the dense arm, on the default Jacobian, leaves 0 at
      another moment than the reference on the FD one, and lands within the
      bound of neither side: only the tighter arm takes one, and the verdict
      is the path's. 338 at 61 and 81 points on arm64, at 99 on x86-64.
    * Broken downward, the laws carry XIIa to its pole at -1 and the dense
      reference is refused (#135; test_clamp_far_below_zero.py): nothing to
      attribute, and the test skips. 338 at this window on x86-64; 339 at
      4001 points on arm64.

    What holds on every face is asserted: KLU's tighter arm keeps the exact
    solution, on the path's side; no arm that leaves it takes that side; the
    verdict follows from the sides. A test of one face at one grid reads the
    coin, not the code, as 338's did: it passed on arm64 and failed on x86-64."""
    path = _B1304[model]
    if not path.exists():
        pytest.skip(f"rr_parity corpus model {model} not present")
    job = D.Job(model, path, 0.0, 2.0, 101, rtol=1e-6, atol=1e-9)
    v = D.check_all(job, None, ("klu_vs_dense",))["klu_vs_dense"]
    if v.status == "exception":
        # The note is the first line of whichever run failed. Skip only where
        # KLU completes and the dense run is the refusal at XIIa's pole.
        D.run(job, force_sparse_linear_solver=True, jacobian="fd")
        with pytest.raises(bngsim.SimulationError, match="carried concentration 'XIIa'"):
            D.run(job, force_dense_linear_solver=True, jacobian="fd")
        pytest.skip("the dense reference was refused at XIIa's pole here, so nothing to attribute")
    assert v.status == "diff", v
    a = v.extra["attribution"]
    sides = {arm: s["side"] for arm, s in a["arms"].items()}
    assert sides.pop("tighter") == "path", a
    assert "dense" in sides and set(sides.values()) <= {"reference", "neither"}, a
    assert a["verdict"] == ("split" if "reference" in sides.values() else "path"), a


def test_a_klu_base_takes_the_dense_solver_as_its_other_arm(monkeypatch):
    """The third arms of a base KLU took: its own route at a hundredfold tighter
    tolerance, and the dense solver. Every other attribution test's base is
    dense, so without this the dense arm is read only by #1304's corpus models,
    which CI has none of. A copy of a base that ran dense stands in for one KLU
    took (the cached base is shared): the dense arm reruns it exactly."""
    shared = _base("derived-log10")
    routed = shared.linear_solver
    base = shared.against(shared.x)
    base.linear_solver = 1
    asked = []
    real = D.run

    def spy(job, **kw):
        asked.append((job, kw))
        return real(job, **kw)

    monkeypatch.setattr(D, "run", spy)
    arms = D._arm_runs(base)
    assert set(arms) == {"tighter", "dense"}, sorted(arms)
    (tight, tight_kw), (job, dense_kw) = asked
    assert tight_kw == {} and (tight.rtol, tight.atol) == (
        base.job.rtol / 100,
        base.job.atol / 100,
    )
    assert job == base.job and dense_kw == {"force_dense_linear_solver": True}, asked
    assert D.gap(arms["dense"], base.x, base.job.atol) == 0.0
    assert shared.linear_solver == routed


def test_what_1309_adds_runs_in_what_is_left_of_the_cap(monkeypatch):
    """The pool kills a model's worker at its cap, and every row with it: four of
    the nightly's slowest models read TIMEOUT on every check once ``stable``
    reran them (BIOMD0000000810's jittered rerun ground past 900 s). ``stable``
    runs after every check main runs: with the cap spent there, it is TIMEOUT,
    said why, and each other row is what it is without a cap."""
    import time

    job = SHAPES["triangle-maxabs-narrow"]
    free = D.check_all(job, None, CHECKS)
    real = D.check_backend

    def backend_then_spent(base):
        v = real(base)
        D._DEADLINE = time.monotonic() - 1.0
        return v

    monkeypatch.setattr(D, "check_backend", backend_then_spent)
    heard = []
    bounded = D.check_all(
        job, None, CHECKS, deadline=time.monotonic() + 1e3, on_verdict=lambda c, v: heard.append(c)
    )
    assert bounded["stable"].status == "timeout", bounded["stable"]
    assert "not finished within the model's cap" in bounded["stable"].note
    # What finished before the cap was spent is what it is without one; what
    # comes after (stable, and any check another change adds after main's) is
    # TIMEOUT.
    done = heard[: heard.index("backend_as_asked") + 1]
    assert {c: bounded[c].status for c in done} == {c: free[c].status for c in done}
    assert {bounded[c].status for c in heard[len(done) :]} == {"timeout"}, bounded


def test_an_arm_past_the_deadline_is_left_out(monkeypatch):
    """A DIFF's attribution in a spent cap asks no arm, and says nothing."""
    import time

    spend = []

    def planted(base):
        if spend:
            D._DEADLINE = time.monotonic() - 1.0
        return D.Verdict("diff", 0.5, "planted", ref_x=base.x, path_x=base.x * 1.5)

    monkeypatch.setattr(D, "check_grid", planted)
    job = SHAPES["triangle-maxabs-narrow"]
    free = D.check_all(job, None, ("grid",))["grid"].extra["attribution"]
    assert free["arms"], free
    spend.append(1)
    spent = D.check_all(job, None, ("grid",), deadline=time.monotonic() + 1e3)["grid"]
    assert spent.extra["attribution"] == {"verdict": "unattributed", "arms": {}}, spent.extra


# X' = -k·x·(x - 0.3)·(x - 1), x = X/s (issue #1001's cubic): roots 0 and 1
# stable, 0.3 not. At s = 1e-9 the residual test accepts it one step from x0.
_CUBIC = """begin parameters
 1 k 1.0
 2 s 1e-9
 3 X0 {x0}*s
 4 k3 k/(s*s)
 5 k2 k*1.3/s
 6 k1 k*0.3
end parameters
begin species
 1 X() X0
end species
begin reactions
 1 1,1,1 1,1 k3
 2 1,1 1,1,1 k2
 3 1 0 k1
end reactions
"""


def _steady_job(name, text) -> D.Job:
    path = _DIR / f"{name}.net"
    path.write_text(text)
    return D.Job(name, path, 0.0, 10.0, 21, atol=1e-20)


_STEADY = {
    "cubic": _steady_job("cubic", _CUBIC.format(x0=0.5)),
    "cubic-low": _steady_job("cubic-low", _CUBIC.format(x0=0.2)),
    # X' = k·X·(1 - X): 0 unstable, 1 stable
    "logistic": _steady_job(
        "logistic",
        "begin parameters\n 1 k 1.0\nend parameters\nbegin species\n 1 X() 0.1\nend species\n"
        "begin reactions\n 1 1 1,1 k\n 2 1,1 1 k\nend reactions\n",
    ),
    # A <-> B, and D counting B's flux: D never settles, A and B do (#74)
    "counter": _steady_job(
        "counter",
        "begin parameters\n 1 k 1.0\nend parameters\n"
        "begin species\n 1 A() 1\n 2 B() 0\n 3 D() 0\nend species\n"
        "begin reactions\n 1 1 2 k\n 2 2 1 k\n 3 2 2,3 k\nend reactions\n",
    ),
    # A <-> B: A + B = 1 is kept, at A = B = 0.5
    "exchange": _steady_job(
        "exchange",
        "begin parameters\n 1 k 1.0\nend parameters\n"
        "begin species\n 1 A() 1\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 k\n 2 2 1 k\nend reactions\n",
    ),
    # the Brusselator at a = 1, b = 3: a limit cycle, no state to rest at
    "brusselator": _steady_job(
        "brusselator",
        "begin parameters\n 1 a 1.0\n 2 b 3.0\n 3 one 1.0\nend parameters\n"
        "begin species\n 1 X() 1\n 2 Y() 1\nend species\n"
        "begin reactions\n 1 0 1 a\n 2 1,1,2 1,1,1 one\n 3 1 2 b\n 4 1 0 one\nend reactions\n",
    ),
}


def _plant(monkeypatch, state=None, criterion=None, converged=True):
    """steady_state() as planted: told ``criterion``, or reporting ``converged``
    at ``state(x)``, x the state it reached."""
    real = bngsim.Simulator.steady_state

    def planted(self, **kw):
        ss = real(self, **({**kw, "criterion": criterion} if criterion else kw))
        if state is None:
            return ss
        x = np.asarray(state(np.asarray(ss.concentrations)), dtype=float)
        return types.SimpleNamespace(
            converged=converged,
            concentrations=x,
            species_names=ss.species_names,
            residual=ss.residual,
        )

    monkeypatch.setattr(bngsim.Simulator, "steady_state", planted)


@pytest.mark.parametrize("method", sorted(D.STEADY_METHODS.values()))
def test_steady_state_is_where_a_run_comes_to_rest(method):
    """#1311: both methods, at a scale of 1e-9, against a run from x0 = 0.5."""
    v = D.check_steady(D.Base(_STEADY["cubic"]), method)
    assert v.status == "pass", v


@pytest.mark.parametrize(
    ("method", "why"),
    [
        ("integration", "a run from its state moves 1.0e+00; X() at 5.00022e-10"),
        # a run from the root itself stays: the spectrum is what sees it
        ("newton", "an unstable root (an eigenvalue of 2.10e-01), where the run comes"),
    ],
)
def test_a_converged_that_a_run_leaves_is_a_finding(monkeypatch, method, why):
    """#1001: under the residual test the cubic came back converged one step
    from x0 (integration), and on its unstable root 0.3 (Newton); every run
    ends at 1."""
    _plant(monkeypatch, criterion="residual")
    v = D.check_steady(D.Base(_STEADY["cubic"]), method)
    assert v.status == "diff" and why in v.note, v


def test_a_solve_that_has_not_converged_is_not_judged(monkeypatch):
    """Judged only where both have arrived: the same wrong state, not reported
    converged, is SKIP."""
    base = D.Base(_STEADY["cubic"])
    _plant(monkeypatch, state=lambda x: x * 0.5, converged=False)
    v = D.check_steady(base, "integration")
    assert v.status == "skip" and "did not converge" in v.note, v


def test_another_stable_state_is_recorded_not_a_finding(monkeypatch):
    """A solve that comes to rest on the cubic's other stable root, 0: the
    model's bistability, which the baseline holds (SKIP, both named)."""
    base = D.Base(_STEADY["cubic"])
    _plant(monkeypatch, state=np.zeros_like)
    v = D.check_steady(base, "integration")
    assert v.status == "skip" and "two stable states" in v.note, v
    assert "X() at 0 against 1e-09" in v.note, v.note


def test_a_root_whose_jacobian_is_refused_is_not_judged(monkeypatch):
    """Model.jacobian() refuses a difference quotient whose step decides an
    entry (#1002): the spectrum that tells an unstable root from another
    stable state is not to be had, so the row is SKIP, said why, not an
    EXCEPTION against steady_state()."""
    base = D.Base(_STEADY["cubic"])
    _plant(monkeypatch, state=np.zeros_like)

    def refuses(self, y, t=0.0, **kw):
        raise bngsim.SimulationError(
            "Model.jacobian() does not return this difference quotient (issue #1002)"
        )

    monkeypatch.setattr(bngsim.Model, "jacobian", refuses)
    v = D.check_steady(base, "integration")
    assert v.status == "skip" and "stability is not judged" in v.note, v


def test_a_root_off_the_conserved_total_is_a_finding(monkeypatch):
    """A = B = 1 is a stable root a run rests at, of the system whose A + B is
    2. This one's is 1: the solve has left it."""
    base = D.Base(_STEADY["exchange"])
    _plant(monkeypatch, state=np.ones_like)
    v = D.check_steady(base, "newton")
    assert v.status == "diff" and "conservation law 0 (total 2 against the run's 1)" in v.note, v


def test_an_unstable_root_a_run_does_not_leave_is_a_finding(monkeypatch):
    """The logistic's 0 is a root no run from it leaves, and an unstable one: a
    solve converged there has not found where the system rests."""
    base = D.Base(_STEADY["logistic"])
    _plant(monkeypatch, state=np.zeros_like)
    v = D.check_steady(base, "newton")
    assert v.status == "diff" and "unstable root" in v.note, v


def test_an_initial_state_on_an_unstable_root_is_not_a_finding(monkeypatch):
    """The cubic from x0 = 0.3 exactly: the solve stays on the root it starts
    on, as the trajectory does in exact arithmetic; a run that leaves it on
    rounding is no reference (BIOMD0000000518)."""
    job = _steady_job("cubic-on-root", _CUBIC.format(x0=0.3))
    base = D.Base(job)
    real = D._steady_run

    def rounding(job, state=None, rtol_scale=1.0):
        return real(job, state if state is not None else [0.3e-9 * (1 + 1e-6)], rtol_scale)

    monkeypatch.setattr(D, "_steady_run", rounding)
    _plant(monkeypatch, state=lambda x: np.full_like(x, 0.3e-9))
    v = D.check_steady(base, "integration")
    assert v.status == "skip" and "the initial state is an unstable root" in v.note, v


def test_a_run_to_equilibrium_that_fails_is_no_reference(monkeypatch):
    base = D.Base(_STEADY["cubic"])

    def fails(*a, **kw):
        raise bngsim.SimulationError("CVODE integration failed (CV_CONV_FAILURE)")

    monkeypatch.setattr(D, "_steady_run", fails)
    v = D.check_steady(base, "integration")
    assert v.status == "skip" and "no reference: the run to t=1e+06, or at a tenfold" in v.note, v


def test_a_run_that_has_not_settled_is_not_judged(monkeypatch):
    """A limit cycle has no state for steady_state() to be, converged or not:
    SKIP, said why, never a finding."""
    base = D.Base(_STEADY["brusselator"])
    assert D.check_steady(base, "integration").status == "skip"
    _plant(monkeypatch, state=lambda x: x)
    v = D.check_steady(D.Base(_STEADY["brusselator"]), "integration")
    assert v.status == "skip" and "has not settled" in v.note, v


def test_a_write_only_accumulator_is_left_out_of_both():
    """A counter has no steady value: the solve is asked to settle the rest
    (mask=~is_pure_sink()), and the run is read on the rest."""
    v = D.check_steady(D.Base(_STEADY["counter"]), "integration")
    assert v.status == "pass", v


def test_a_solve_from_another_basin_is_the_run_from_there():
    """The run starts where the solve does: from x0 = 0.2, both end at 0."""
    v = D.check_steady(D.Base(_STEADY["cubic-low"]), "integration")
    assert v.status == "pass", v


def test_a_run_to_equilibrium_past_the_cap_is_a_timeout(monkeypatch):
    """The model's cap reached in the reference run is this check's TIMEOUT, as
    any check's is (#1321), not a missing reference."""
    base = D.Base(_STEADY["cubic"])

    def spent(*a, **kw):
        raise bngsim.SimulationTimeout("the model's cap is spent")

    monkeypatch.setattr(D, "_steady_run", spent)
    v = D.check_steady(base, "integration")
    assert v.status == "timeout" and "the model's cap" in v.note, v
