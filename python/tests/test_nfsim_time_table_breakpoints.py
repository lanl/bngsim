"""method='nf' and a table function indexed by time: read at its breakpoints
(issue #892), and integrated where a linear one ramps (issue #1179).

NFsim's ``stepTo`` drew each waiting time from the total propensity and read
the propensities again only after a reaction fired. A table indexed by time
changes value at its breakpoints whether or not anything fires, so a rate that
is 0 until a later breakpoint never turned on: 100 ``X(p~1)`` with the table at
0 until t = 1 stayed at 100 on every row. With a total propensity next to 0 the
stale value held until the next firing. The fixture as written fires often
enough to hide it, and the test that ran it checked only the result's shape.

The tables are now read at each breakpoint, and a draw that passes one is drawn
again there, which is exact for a step table. A linear table also moves between
its breakpoints, and held at its last reading the run lagged the ramp: from the
same start with the table ramping from 0 at t = 0, all 100 were still there at
t = 1, where the exact mean is 1.75. Where a linear table ramps, the next firing
is now where the integrated propensity reaches an Exp(1) draw.

Each start is checked against the exact mean, as ``test_rulemonkey_model_time.py``
checks method='rm', on outputs at the breakpoints and between them: the step
table's in closed form, a linear table's by an ODE solve of one molecule's
probability, which every molecule follows on its own. The helpers are this
file's own: the Windows NFsim leg runs on edits to ``test_nfsim*.py`` only, so
an oracle imported from another test file could change without running it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import bngsim
import numpy as np
import pytest
from scipy.integrate import solve_ivp

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
TFUN = DATA / "nfsim" / "tfun_new_format" / "valid_inline_step_time.xml"
TFUN_FILE = DATA / "nfsim" / "tfun_new_format" / "valid_file_step_time.xml"
LINEAR = DATA / "nfsim" / "tfun_new_format" / "valid_inline_linear_time.xml"
LINEAR_FILE = DATA / "nfsim" / "tfun_new_format" / "valid_file_linear_time.xml"

# (from, to, rate back to X(p~0)) for the fixture's table: 0, 10 from t = 1, 20 from 2.
STEPS = ((0.0, 1.0, 0.0), (1.0, 2.0, 10.0), (2.0, math.inf, 20.0))

# t = 1 and 2, the table's breakpoints, are outputs in the first grid. The second
# grid puts each one inside an output interval, where a draw past it is dropped.
ON_BREAKPOINTS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
BETWEEN_BREAKPOINTS = [0.0, 0.75, 1.5, 2.25, 3.0]

# The same grids for the linear table, with t = 0.25 added: from the start where
# nothing can fire, the held rate was furthest off there and at t = 0.5.
RAMP_ON_BREAKPOINTS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
RAMP_BETWEEN_BREAKPOINTS = [0.0, 0.25, 0.5, 0.75, 1.25, 2.25, 3.0]

STARTS = [
    (0, 100, 0.1),  # nothing can fire until the table turns on
    (1, 99, 0.001),  # next to nothing can fire until then
    (1, 99, 1.0),
    (100, 0, 0.1),  # the fixture as written
]


def _has_nfsim() -> bool:
    try:
        from bngsim._bngsim_core import HAS_NFSIM

        return bool(HAS_NFSIM)
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")


def _edit(tmp_path: Path, src: Path, *edits: tuple[str, str]) -> Path:
    xml = src.read_text(encoding="utf-8")
    for old, new in edits:
        assert old in xml, old
        xml = xml.replace(old, new)
    out = tmp_path / "edited.xml"
    out.write_text(xml, encoding="utf-8")
    return out


def _start(n0: int, n1: int, dephos: float | None = None) -> tuple[tuple[str, str], ...]:
    edits = (
        ('concentration="100" name="X(p~0)"', f'concentration="{n0}" name="X(p~0)"'),
        ('concentration="0" name="X(p~1)"', f'concentration="{n1}" name="X(p~1)"'),
    )
    if dephos is not None:
        edits += (
            (
                'id="dephos" type="Constant" value="0.1"',
                f'id="dephos" type="Constant" value="{dephos}"',
            ),
        )
    return edits


def _phos(start: int, dephos: float, t: float, steps=STEPS) -> float:
    """P(X(p~1) at t) for one molecule: X(p~0) -> X(p~1) at dephos, and back at
    the step table's rate, each piece solved."""
    p = float(start)
    for lo, hi, back in steps:
        if t <= lo:
            break
        rate = dephos + back
        eq = dephos / rate
        p = eq + (p - eq) * math.exp(-rate * (min(t, hi) - lo))
    return p


def _ramp(t: float) -> float:
    """The linear table's rate: 0, 10, 20 at t = 0, 1, 2, held at 20 after."""
    return float(np.interp(t, [0.0, 1.0, 2.0], [0.0, 10.0, 20.0]))


def _phos_ode(
    start: int, dephos: float, times: list[float], back: Callable[[float], float] = _ramp
) -> np.ndarray:
    """P(X(p~1)) at *times* for one molecule: X(p~0) -> X(p~1) at dephos, and back
    at back(t), solved between the table's breakpoints so no step crosses a kink."""
    edges = sorted({0.0, 1.0, 2.0, *times})
    p = {0.0: float(start)}
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sol = solve_ivp(
            lambda t, y: dephos * (1.0 - y) - back(t) * y,
            (lo, hi),
            [p[lo]],
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        assert sol.success, sol.message
        p[hi] = float(sol.y[0, -1])
    return np.array([p[t] for t in times])


def _assert_mean(samples: np.ndarray, mean: np.ndarray, var: np.ndarray) -> None:
    """Each column of *samples* (replicates x rows) has the exact *mean*: equal to
    it where the count is certain, within 4.5 standard errors elsewhere."""
    certain = var <= 0.0
    np.testing.assert_array_equal(
        samples[:, certain], np.broadcast_to(mean[certain], samples[:, certain].shape)
    )
    se = np.sqrt(var[~certain] / samples.shape[0])
    z = (samples[:, ~certain].mean(axis=0) - mean[~certain]) / se
    assert np.all(np.abs(z) < 4.5), z


def _assert_ode_mean(
    runs: np.ndarray, n0: int, n1: int, dephos: float, times: list[float], back=_ramp
) -> None:
    q1 = _phos_ode(1, dephos, times, back)
    q0 = _phos_ode(0, dephos, times, back)
    _assert_mean(runs, n1 * q1 + n0 * q0, n1 * q1 * (1 - q1) + n0 * q0 * (1 - q0))


def _nf(xml: Path, seed: int, times: list[float]) -> np.ndarray:
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    sim = bngsim.Simulator(m, method="nf", xml_path=str(xml))
    return np.asarray(sim.run(sample_times=times, seed=seed).observables["X_phos"], dtype=float)


@pytest.mark.parametrize("T", [ON_BREAKPOINTS, BETWEEN_BREAKPOINTS], ids=["on", "between"])
@pytest.mark.parametrize("n0, n1, dephos", STARTS)
def test_a_time_indexed_table_function_matches_the_closed_form(tmp_path, n0, n1, dephos, T):
    xml = _edit(tmp_path, TFUN, *_start(n0, n1, dephos))
    runs = np.array([_nf(xml, seed, T) for seed in range(1, 201)])
    q1 = np.array([_phos(1, dephos, t) for t in T])
    q0 = np.array([_phos(0, dephos, t) for t in T])
    _assert_mean(runs, n1 * q1 + n0 * q0, n1 * q1 * (1 - q1) + n0 * q0 * (1 - q0))


@pytest.mark.parametrize(
    "rows, steps",
    [
        ("0 0\n1 10\n2 20\n", STEPS),
        # NFsim reads a table listed in decreasing time from the other side: 0 at
        # t = 0, 10 on (0, 1], 20 after. At each x it still gives the value from
        # before it, so it is read one ulp past each x instead; read at x, the
        # run kept 0 until t = 1 and 10 until t = 2.
        ("2 20\n1 10\n0 0\n", ((0.0, 1.0, 10.0), (1.0, math.inf, 20.0))),
    ],
    ids=["increasing", "decreasing"],
)
def test_a_time_indexed_table_from_a_file_matches_the_closed_form(tmp_path, rows, steps):
    (tmp_path / "table.tfun").write_text("# time tfun_rate\n" + rows, encoding="utf-8")
    xml = _edit(
        tmp_path, TFUN_FILE, ('file="time_step.tfun"', 'file="table.tfun"'), *_start(0, 100)
    )
    runs = np.array([_nf(xml, seed, ON_BREAKPOINTS) for seed in range(1, 201)])
    q = np.array([_phos(1, 0.1, t, steps) for t in ON_BREAKPOINTS])
    _assert_mean(runs, 100 * q, 100 * q * (1 - q))


@pytest.mark.parametrize(
    "T", [RAMP_ON_BREAKPOINTS, RAMP_BETWEEN_BREAKPOINTS], ids=["on", "between"]
)
@pytest.mark.parametrize("n0, n1, dephos", STARTS)
def test_a_linear_time_table_matches_the_ode(tmp_path, n0, n1, dephos, T):
    # From the first start the held rate kept all 100 to t = 1, against 73.4 at
    # t = 0.25 and 1.75 at t = 1; as written it was 3 SE high at t = 0.5.
    xml = _edit(tmp_path, LINEAR, *_start(n0, n1, dephos))
    runs = np.array([_nf(xml, seed, T) for seed in range(1, 201)])
    _assert_ode_mean(runs, n0, n1, dephos, T)


@pytest.mark.parametrize(
    "rows", ["0 0\n1 10\n2 20\n", "2 20\n1 10\n0 0\n"], ids=["increasing", "decreasing"]
)
def test_a_linear_time_table_from_a_file_matches_the_ode(tmp_path, rows):
    # Either order gives the same ramp. A table listed in decreasing time is read
    # one ulp past each x, and its ramp must still be found between them.
    (tmp_path / "table.tfun").write_text("# time tfun_rate\n" + rows, encoding="utf-8")
    xml = _edit(
        tmp_path, LINEAR_FILE, ('file="time_linear.tfun"', 'file="table.tfun"'), *_start(0, 100)
    )
    runs = np.array([_nf(xml, seed, RAMP_BETWEEN_BREAKPOINTS) for seed in range(1, 201)])
    _assert_ode_mean(runs, 0, 100, 0.1, RAMP_BETWEEN_BREAKPOINTS)


def test_a_rate_law_that_reads_a_linear_table_through_a_function_matches_the_ode(tmp_path):
    # A saturating function of the table: not a polynomial in t, so a panel's
    # quadrature is not exact and its error control has to work. It reads the
    # table through another function, which NFsim builds as a composite.
    tag = '<Function id="tfun_rate" type="TFUN"'
    xml = _edit(
        tmp_path,
        LINEAR,
        *_start(0, 100),
        (
            '<RateLaw id="RR2_RateLaw" type="Function" name="tfun_rate">',
            '<RateLaw id="RR2_RateLaw" type="Function" name="g">',
        ),
        ('<RateConstant value="tfun_rate"/>', '<RateConstant value="g"/>'),
        (
            tag,
            '<Function id="g"><ListOfReferences><Reference name="tfun_rate" type="Function"/>'
            "</ListOfReferences><Expression>20*tfun_rate/(5+tfun_rate)</Expression></Function>"
            "\n      " + tag,
        ),
    )
    runs = np.array([_nf(xml, seed, RAMP_BETWEEN_BREAKPOINTS) for seed in range(1, 201)])
    _assert_ode_mean(
        runs,
        0,
        100,
        0.1,
        RAMP_BETWEEN_BREAKPOINTS,
        back=lambda t: 20 * _ramp(t) / (5 + _ramp(t)),
    )


def test_a_session_run_in_segments_follows_the_ramp(tmp_path):
    # Each segment ends inside the ramp, and what it leaves of its draw is the
    # next one's.
    xml = _edit(tmp_path, LINEAR, *_start(0, 100))
    runs = []
    for seed in range(1, 201):
        with bngsim.NfsimSession(str(xml)) as s:
            s.initialize(seed=seed)
            a = np.asarray(s.simulate(0.0, 0.5, n_points=3).observables["X_phos"])
            b = np.asarray(s.simulate(0.5, 3.0, n_points=6).observables["X_phos"])
        runs.append(np.concatenate([a, b[1:]]))
    T = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    _assert_ode_mean(np.array(runs, dtype=float), 0, 100, 0.1, T)


def test_a_step_table_that_jumps_inside_the_ramp_matches_the_ode(tmp_path):
    # Its jump at t = 0.5 is a breakpoint inside the linear table's ramp, so the
    # ramp is integrated up to it and from it, and the panel ending there reads
    # the step on the side it integrates.
    tag = '<Function id="tfun_rate" type="TFUN"'
    xml = _edit(
        tmp_path,
        LINEAR,
        *_start(0, 100),
        (
            '<RateLaw id="RR2_RateLaw" type="Function" name="tfun_rate">',
            '<RateLaw id="RR2_RateLaw" type="Function" name="g">',
        ),
        ('<RateConstant value="tfun_rate"/>', '<RateConstant value="g"/>'),
        (
            tag,
            '<Function id="gate" type="TFUN" mode="inline" ctrName="time" xData="0,0.5" '
            'yData="0.5,1" method="step"><Expression>__TFUN_VAL__</Expression>'
            "<ListOfReferences/></Function>\n      "
            '<Function id="g"><ListOfReferences><Reference name="tfun_rate" type="Function"/>'
            '<Reference name="gate" type="Function"/></ListOfReferences>'
            "<Expression>tfun_rate*gate</Expression></Function>\n      " + tag,
        ),
    )
    T = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0]
    runs = np.array([_nf(xml, seed, T) for seed in range(1, 201)])
    _assert_ode_mean(runs, 0, 100, 0.1, T, back=lambda t: _ramp(t) * (0.5 if t < 0.5 else 1.0))
