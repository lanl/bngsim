"""Where a condition on the clock changes value, found by evaluating it.

The crossing-stop resolvers in :mod:`bngsim._switch_sensitivity` each read one
shape exactly: a residual linear in time, a repeating schedule, a step of an
affine argument, a sinusoid, a polynomial. A condition of any other shape got
no stop, and a window narrower than an integrator step was stepped over with
nothing said (issues #899, #904): ``abs(time() - 120) < 0.5``, ``trunc`` and
``round`` and ``sign``, a step of ``time()^2``, a ``mod`` whose threshold is
itself a ``mod`` of constants.

This module is what is asked when none of them reads a condition. It does not
parse the condition's arithmetic. It evaluates it, with the model's own
evaluator (``NetworkModel.evaluate_over_times``), which is the arithmetic that
decides the double at which the rate law changes branch, and looks for where
the value changes:

1. Every step call in the text (``floor``, ``ceil``, ``trunc``, ``round``,
   ``rint``, ``frac``, ``sign``, ``mod``, ...) jumps where its argument crosses
   a level, an integer for most. The argument is evaluated on a grid over the
   window, each level it passes between two grid points is bisected to adjacent
   doubles, and an extremum of the argument between grid points is searched
   for, so a level touched and left again inside one cell is found too. The
   call's own value is then bisected around each crossing, since the call need
   not jump on the double its argument reaches the level at: ``ceil`` jumps
   where the argument first exceeds it, and ``mod`` is C's exact fmod of a
   quotient this only sees rounded. Inner calls are read first, and their jumps
   become grid points for the calls that hold them.
2. A call that bends (``abs``, ``min``, ``max``, ``clamp``) is read the same
   way for where it turns, and so is where a text stops being a number (the
   edge of a ``sqrt`` or a ``ln``). Those are grid points too.
3. A comparison's two sides are read against each other, with a point between
   every two of the points above, so each smooth piece has one inside it.
4. The condition itself is then evaluated at every point and on the few
   doubles around every candidate, and each change between two points is
   bisected.

What comes back is, for each change, the first double at which the evaluator
reads the new value, which is where a stop has to sit for the restart to begin
on the branch that starts there (issue #891).

The whole of it is done on a grid and again on one twice as fine, until both
find the same number of changes (as instants: changes a billionth of the
window apart are one). A grid is an even one, which lands on the
round numbers conditions are written with, and as many points again at
spacings no period divides, so that an oscillation faster than the grid is
not read at one phase by every pass. A condition that does not settle by two
million cells is not resolved, and the caller is told so rather than handed
what the last grid happened to catch.

A step call whose argument is within its own rounding of a level it crosses,
and changes side of it time and again over more than an instant of the window
(a billionth of it), is not resolved either (:class:`Flickers`): `sign` of a
power of ``time() - 5`` written out term by term has no time at which it
changes. A flicker shorter than that, which any sum of terms can show around
a crossing, is one crossing.

Where an argument does not get clear of its rounding within a few instants of
a crossing and does not flicker either (a small sum on a large one moves by a
few of its own last places in all, and is a staircase), the call is not read
by what it is at the two ends of a bracket. It is read from one end to the
other, a quarter of an instant at a time, and as far out as the argument is
within its rounding of a level. Where that is more of the window than can be
read (as many points as the finest grid has cells), the call is refused the
same way. A staircase that calls and comparisons inside the argument make is
none of this: where they jump is known, and what the argument does there is
their jump.

What it cannot see is a window narrower than a grid cell that no sample has
any sign of: a smooth residual that is flat to the last bit everywhere but
inside the window (a Gaussian a thousandth of a cell wide), or a window made
by a discontinuity this does not know the levels of (ExprTk's ``%`` operator,
``iclamp``). A residual that turns inside the cell where the samples around it
lean toward the turn, the calls of 1 and 2, and a window at the edge of a
domain are read at any width.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable, Collection, Sequence
from typing import NamedTuple

import numpy as np

# Grid cells of the first pass over the window, and of the last one tried. A
# point is added between every two, and the text is evaluated a few times at
# each, at about 15 ns: a millisecond or two for the first two passes, which
# are all a condition that settles at once costs, once per window and
# parameter point.
GRID_CELLS = 1 << 12
MAX_GRID_CELLS = 1 << 21
# Doubles on each side of a candidate that the condition is evaluated on.
_NEIGHBOURS = 3
# Changes of a condition this many ulps apart or less are one instant. An even
# number of them is a window with no measure, which is not a stop unless an
# event trigger reads it (two restarts that close held a large jump between
# them, issue #714). An odd number is one change, and the rounding of the
# comparison flickering around it.
TOUCH_ULPS = 64.0
# Golden-section iterations for an extremum between grid points: the bracket
# shrinks by 0.618 a step, so 80 take it from a cell to below an ulp.
_GOLDEN_STEPS = 80
_INVPHI = (math.sqrt(5.0) - 1.0) / 2.0
# The points a stretch with a sample at one end only is read at, as parts of
# it from that end: each half the one after, down to a sixteen-millionth.
_END_LADDER = 0.5 ** np.arange(24, 0, -1)

Evaluate = Callable[[str, np.ndarray], np.ndarray]
Sampled = Callable[[np.ndarray], np.ndarray]


class OverBudget(Exception):
    """The condition changes value more times in the window than the caller
    places stops for. Carries how many candidates were counted."""

    def __init__(self, count: int):
        super().__init__(count)
        self.count = int(count)


class Unresolved(OverBudget):
    """The number of changes found did not settle as the grid was refined: the
    condition turns faster than two million cells over the window can follow."""


class Flickers(Unresolved):
    """The argument of a step call is within its own rounding of a level it
    crosses, and changes side of that level time and again over more than an
    instant of the run: between ``lo`` and ``hi``, for ``call`` (the first
    such stretch of ``count``). Where the call changes is then not a time,
    and no stop is placed for it.

    With ``unread``, it does not change side time and again, but is within
    its rounding of the level for more of the run than the call can be read
    over for every step it takes across it (``lo`` to ``hi`` is the longest
    such stretch of ``count``).

    With ``coarse``, the call is a sawtooth (``mod``, ``frac``) and the whole
    number it takes off its argument is beyond :data:`_SAWTOOTH_AT_MOST`
    somewhere between ``lo`` and ``hi``, the whole run: what is left is the
    rounding of the argument, a few steps to the wrap, and no sawtooth."""

    def __init__(
        self,
        call: str,
        lo: float,
        hi: float,
        count: int = 1,
        unread: bool = False,
        coarse: bool = False,
    ):
        super().__init__(count)
        self.call = call
        self.lo = float(lo)
        self.hi = float(hi)
        self.unread = unread
        self.coarse = coarse


class Phase(NamedTuple):
    """One argument of one call whose level crossings are read."""

    call: str  # the call's own text
    phase: str  # what crosses a level
    zero_only: bool  # zero is the one level; every integer otherwise
    bends: bool  # the call turns there (abs, min, max, clamp); it jumps otherwise
    # The two texts the phase is the difference of, where the call compares
    # two of its arguments (`min`, `max`, `clamp`, `inrange`) or adds a number
    # to one (`rint`): the evaluator evaluates each and compares them, and
    # each is evaluated on its own here. Handed
    # `((3)-(2.9999999999999991 + 0.1*(time()-100)^2))` whole it folds the
    # constants, and that difference is zero 1.2e-8 outside the window `min`
    # of the two is below 3 in.
    sides: tuple[str, str] | None = None


# Step calls and the argument whose level crossings are their jumps. Each entry
# maps the call's arguments to ``(phase text, zero_only)``: the call jumps where
# the phase crosses an integer, or zero alone.
#
# A phase is written so that the call is `floor` of it, up to a sign: the call
# then changes where the phase first REACHES a whole number, on a plateau of
# the argument as at a crossing, and has no level where it has no jump. Read
# off the bare argument, `trunc`, `frac` and `mod` had a level at zero, which
# none of them jumps at (`frac(time()^2 - 10*time() + 25)` was refused for the
# flicker of its argument there), and `round` of an argument that sits on -0.5
# and leaves it had no crossing anywhere near where it changes.
def _phase_first(args: Sequence[str]) -> list[tuple[str, bool]]:
    return [(f"({args[0]})", False)]


def _phase_ceil(args: Sequence[str]) -> list[tuple[str, bool]]:
    # ceil(x) is -floor(-x).
    return [(f"(-({args[0]}))", False)]


def _phase_size(args: Sequence[str]) -> list[tuple[str, bool]]:
    # trunc(x) is sign(x)*floor(abs(x)), and frac(x) is x less that.
    return [(f"(abs({args[0]}))", False)]


def _phase_half(args: Sequence[str]) -> list[tuple]:
    # The evaluator's rint is floor(x + 0.5): x evaluated, and a half added to
    # that. The argument is evaluated on its own here and the half added to
    # what comes back. Handed `((0.4999999999999 + 1e-15*time())+0.5)` whole,
    # the evaluator adds the two constants first, and that sum reaches 1 on
    # another double than the call's does: 0.03 before `rint` changed in a
    # window of 11, with no stop for the change where the window began within
    # a few doubles of the level.
    return [(f"(({args[0]})+0.5)", False, (args[0], "-0.5"))]


def _phase_round(args: Sequence[str]) -> list[tuple]:
    # The evaluator's round is C's (issue #684): a half away from zero, with
    # no sum that can itself round. It changes where abs(x) reaches a whole
    # number and a half, which is where abs(x) less a half reaches a whole
    # number, and that difference of two doubles is one: abs(x) is evaluated
    # and the half taken off here. (Written `abs(x) + 0.5`, as it was read
    # while the evaluator's round was floor(x + 0.5), the sum is 1 on the
    # double under a half, where `round` is still 0.) From 2^52 on every
    # double is a whole number and `round` changes with each: the half is not
    # taken off there (:func:`_less_a_half`).
    return [(f"(abs({args[0]})-0.5)", False, (f"abs({args[0]})", "0.5"))]


def _less_a_half(size: np.ndarray) -> np.ndarray:
    """The phase of ``round`` from the size of its argument: less a half, and
    from 2^52 on the size itself.

    Below 2^52 the difference is exact. From there every ``size - 0.5`` is a
    tie, rounded to the even neighbour, and `floor` of it changes on every
    other double where C's ``round`` changes on each:
    ``round(4503599627370498 + 0.9*exp(-((time() - 50)/0.5)^2))`` is one more
    for 0.77 of time, and had no crossing, no stop and no word.
    """
    return np.where(size < 2.0**52, size - 0.5, size)


def _phase_of_round(ev: Evaluate, sized: str) -> Sampled:
    """What ``round`` crosses its levels with, over times: *sized*, the size of
    its argument as the evaluator has it, less a half (:func:`_less_a_half`)."""

    def phase(at: np.ndarray) -> np.ndarray:
        return _less_a_half(ev(f"({sized})", at))

    return phase


def _phase_sign(args: Sequence[str]) -> list[tuple[str, bool]]:
    return [(f"({args[0]})", True)]


def _phase_mod(args: Sequence[str]) -> list[tuple[str, bool]]:
    return [(f"(abs(({args[0]})/({args[1]})))", False)]


def _phase_roundn(args: Sequence[str]) -> list[tuple[str, bool]]:
    # The evaluator rounds to a whole number of digits from 0 to 16, whatever
    # is asked for: `roundn(x, -1)` is `round(x)`. As `round`, away from zero.
    digits = f"max(0,min(16,floor({args[1]})))"
    return [(f"(abs({args[0]})*10^({digits})+0.5)", False)]


def _difference(a: str, b: str) -> tuple[str, bool, tuple[str, str]]:
    return (f"(({a})-({b}))", True, (a, b))


def _phase_inrange(args: Sequence[str]) -> list[tuple]:
    return [_difference(args[1], args[0]), _difference(args[1], args[2])]


_PHASES: dict[str, tuple[int, Callable[[Sequence[str]], list[tuple]]]] = {
    "floor": (1, _phase_first),
    "ceil": (1, _phase_ceil),
    "frac": (1, _phase_size),
    "trunc": (1, _phase_size),
    "round": (1, _phase_round),
    "rint": (1, _phase_half),
    "sign": (1, _phase_sign),
    "sgn": (1, _phase_sign),
    "mod": (2, _phase_mod),
    "roundn": (2, _phase_roundn),
    "inrange": (3, _phase_inrange),
}


# Calls that bend where a step call jumps: the value is continuous there and
# its slope is not. A pulse is written with these alone, as
# `max(0, 1 - abs(time() - t0)/w)`, and is flat on both sides of itself.
def _phase_pair(args: Sequence[str]) -> list[tuple]:
    return [_difference(a, b) for i, a in enumerate(args) for b in args[i + 1 :]]


def _phase_clamp(args: Sequence[str]) -> list[tuple]:
    return [_difference(args[1], args[0]), _difference(args[1], args[2])]


_BENDS: dict[str, tuple[tuple[int, ...], Callable[[Sequence[str]], list[tuple]]]] = {
    "abs": ((1,), _phase_sign),
    "max": ((2, 3), _phase_pair),
    "min": ((2, 3), _phase_pair),
    "clamp": ((3,), _phase_clamp),
}
_CALL_NAME = re.compile(r"(?<![A-Za-z0-9_.])(" + "|".join([*_PHASES, *_BENDS]) + r")\s*\(")


def _close_paren(text: str, open_paren: int) -> int:
    depth = 0
    for i in range(open_paren, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_commas(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


_RELATIONAL = re.compile(r"<=|>=|==|!=|<|>")


def _comparisons(text: str) -> list[str]:
    """Every comparison in *text*, each as a text of its own: from the
    parenthesis or comma that opens the stretch its operator stands in to the
    one that closes it. (Two joined by ``and`` or ``or`` in one stretch are
    one text, which is true or false like either.)"""
    found: list[str] = []
    for m in _RELATIONAL.finditer(text):
        depth, begin = 0, m.start() - 1
        while begin >= 0:
            c = text[begin]
            if c == ")":
                depth += 1
            elif c == "(":
                if depth == 0:
                    break
                depth -= 1
            elif c == "," and depth == 0:
                break
            begin -= 1
        depth, end = 0, m.end()
        while end < len(text):
            c = text[end]
            if c == "(":
                depth += 1
            elif c == ")":
                if depth == 0:
                    break
                depth -= 1
            elif c == "," and depth == 0:
                break
            end += 1
        compared = text[begin + 1 : end].strip()
        if compared and compared not in found:
            found.append(compared)
    return found


def phases(text: str) -> list[Phase]:
    """Every step call and every bending call in *text* with the argument
    whose level crossings are read for it, innermost first, without repeats."""
    found: list[tuple[int, Phase]] = []  # (calls nested inside, phase)

    def walk(segment: str) -> int:
        """Record the calls in *segment*; the deepest nesting found in it."""
        deepest = 0
        cursor = 0
        while True:
            m = _CALL_NAME.search(segment, cursor)
            if m is None:
                return deepest
            open_paren = m.end() - 1
            close = _close_paren(segment, open_paren)
            if close < 0:
                return deepest
            inner = segment[open_paren + 1 : close]
            below = walk(inner)
            args = [a.strip() for a in _split_commas(inner)]
            name = m.group(1)
            if name in _PHASES:
                arity, phase_of = _PHASES[name]
                arities: tuple[int, ...] = (arity,)
            else:
                arities, phase_of = _BENDS[name]
            if len(args) in arities and all(args):
                call = segment[m.start() : close + 1]
                for phase, zero_only, *sides in phase_of(args):
                    found.append((below, Phase(call, phase, zero_only, name in _BENDS, *sides)))
            deepest = max(deepest, below + 1)
            cursor = close + 1

    walk(text)
    found.sort(key=lambda entry: entry[0])
    out: list[Phase] = []
    for _, phase in found:
        if phase not in out:
            out.append(phase)
    return out


def knows_steps_of(text: str) -> bool:
    """Whether *text* holds a step call whose jumps are read from a phase."""
    return any(not phase.bends for phase in phases(text))


def _merge(nodes: np.ndarray, more: np.ndarray) -> np.ndarray:
    """The sorted union of *nodes*, which is sorted and has no repeats, with a
    few more points. The many are not sorted again for the few: sorting the
    grid once per added handful was most of what a scan with nothing to find
    cost.
    """
    if more.size == 0:
        return nodes
    if more.size * 8 > nodes.size:
        return np.union1d(nodes, more)
    more = np.unique(more)
    at = np.searchsorted(nodes, more)
    new = (at >= nodes.size) | (nodes[np.minimum(at, nodes.size - 1)] != more)
    return np.insert(nodes, at[new], more[new])


def _adjacent(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.nextafter(lo, hi) >= hi


# Points a round of :func:`_bisect` may evaluate. With one bracket a round cuts
# it in 64, and seven rounds do what forty-one halvings did, at a fifth of the
# calls into the evaluator, which is what a scan with a change or two costs.
_SECTION_POINTS = 4096
_SECTIONS = 64


def _bisect(
    side_of: Callable[[np.ndarray, np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    side_lo: np.ndarray,
) -> np.ndarray:
    """For each bracket, the first double after ``lo`` whose side differs from
    ``lo``'s on the path the search takes. ``side_of(times, which)`` gives the
    side of each time, *which* being the brackets' indices.

    A round cuts each live bracket into as many sections as the points allow
    and keeps the first whose ends are on different sides, then halves what is
    left once, which is what moves a bracket a few doubles wide.
    """
    lo = lo.copy()
    hi = hi.copy()
    live = np.nonzero(~_adjacent(lo, hi))[0]
    for _ in range(2200):  # the spacing of doubles halves at most ~2100 times
        if live.size == 0:
            break
        k = min(_SECTIONS, _SECTION_POINTS // live.size)
        if k > 2:
            a, b = lo[live], hi[live]
            cuts = a[:, None] + (b - a)[:, None] * (np.arange(1, k) / k)[None, :]
            same = (
                side_of(cuts.ravel(), np.repeat(live, k - 1)).reshape(live.size, k - 1)
                == side_lo[live][:, None]
            )
            # The first cut on the other side, or none of them.
            none = same.all(axis=1)
            j = np.where(none, k - 1, np.argmin(same, axis=1))
            rows = np.arange(live.size)
            hi[live] = np.where(none, b, cuts[rows, np.minimum(j, k - 2)])
            lo[live] = np.where(j == 0, a, cuts[rows, np.maximum(j - 1, 0)])
            # Rounding can put a cut on an end; the halving below still moves.
            lo[live] = np.maximum(lo[live], a)
            hi[live] = np.minimum(hi[live], b)
            live = live[~_adjacent(lo[live], hi[live]) & (hi[live] > lo[live])]
            if live.size == 0:
                break
        mid = lo[live] + 0.5 * (hi[live] - lo[live])
        # Rounding can put the midpoint on an end where they are two apart.
        mid = np.where(
            (mid <= lo[live]) | (mid >= hi[live]), np.nextafter(lo[live], hi[live]), mid
        )
        same = side_of(mid, live) == side_lo[live]
        lo[live[same]] = mid[same]
        hi[live[~same]] = mid[~same]
        live = live[~_adjacent(lo[live], hi[live])]
    return hi


def _ladder_turns(
    f: Sampled, near: np.ndarray, u_near: np.ndarray, far: np.ndarray, u_far: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where *f* turns between each *near* point and its *far* one, as far as a
    ladder of points that halve the distance to *near* shows: ``(a, b, sign)``,
    an interval with a turn in it and -1 for a maximum, 1 for a minimum.

    For a stretch with a sample at one end only that says anything of the
    other: beside an end of the window, where no sample lies beyond, and
    beside a point of the grid with the doubles around it, which is where
    something jumps or bends and what is read on its far side is another
    piece of the function. Read at the middle of such a stretch a function
    says whether it turns half a cell away, and nothing of a turn a millionth
    of a cell from the point; read on the ladder it says both, in one call of
    the evaluator for all of them.
    """
    none = np.empty(0)
    if near.size == 0:
        return none, none, none
    rungs = near[:, None] + (far - near)[:, None] * _END_LADDER[None, :]
    read = f(rungs.ravel()).reshape(rungs.shape)
    # In order from near to far, with the two ends. (Rungs that round onto an
    # end, where the stretch is a few million doubles, repeat what it reads.)
    at = np.concatenate((near[:, None], rungs, far[:, None]), axis=1)
    was = np.concatenate((u_near[:, None], read, u_far[:, None]), axis=1)
    # It moves one way and then the other, with any number of points that read
    # the same between: from the point before the first move to the point
    # after the second. (A stretch that reads the same and then moves is no
    # turn: `time()^2 - 2000` is -2000 on every rung up to 1e-7. A move to or
    # from what is not a number is a move either way.)
    with np.errstate(invalid="ignore"):
        way = np.sign(np.diff(was, axis=1))
    moved = way != 0.0
    column = np.arange(way.shape[1])[None, :]
    last = np.maximum.accumulate(np.where(moved, column, -1), axis=1)
    before = np.concatenate((np.full((near.size, 1), -1), last[:, :-1]), axis=1)
    turned = (
        moved & (before >= 0) & (way != np.take_along_axis(way, np.maximum(before, 0), axis=1))
    )
    row, second = np.nonzero(turned)
    first = before[row, second]
    if row.size > 1:
        # One turn of each stretch: the one it moves furthest into and out of.
        # A function that leans toward a turn has one. Rounding noise has one
        # at every other rung, and each was searched and left a point in the
        # noise, at which the call was then found to flip once more: 619 stops
        # at the two edges of a window where 236 had been found without.
        with np.errstate(invalid="ignore"):
            size = np.abs(np.diff(was, axis=1))
        deep = np.minimum(size[row, first], size[row, second])
        deep = np.where(np.isnan(deep), np.inf, deep)
        order = np.lexsort((-deep, row))
        keep = order[np.concatenate(([True], row[order][1:] != row[order][:-1]))]
        row, first, second = row[keep], first[keep], second[keep]
    lo, hi = at[row, first], at[row, second + 1]
    return np.minimum(lo, hi), np.maximum(lo, hi), np.where(way[row, first] > 0.0, -1.0, 1.0)


def _least(f: Sampled, sign: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each interval ``[a, b]``, a point at which ``sign * f`` is least, of
    those it was read at.

    (Not the middle of where the search ends. On a plateau two points read
    the same and the search goes on to one side of it: at two of the fifteen
    tops of `cos(0.001*t) + 2`, each 3 for 4.7e-5, it ended a few doubles past
    the plateau's edge, with no point on the plateau to show it.)
    """
    sections = min(_SECTIONS, _SECTION_POINTS // a.size)
    if sections > 2:
        # Few to search: each is cut into sections and the two around its least
        # reading are kept, which takes a cell to a few doubles in eight calls
        # of the evaluator where the steps below take eighty.
        rows = np.arange(a.size)
        parts = (np.arange(sections + 1) / sections)[None, :]
        best, f_best = a.copy(), np.full(a.size, np.inf)
        for _ in range(_GOLDEN_STEPS):
            cuts = a[:, None] + (b - a)[:, None] * parts
            cuts[:, -1] = b
            read = sign[:, None] * f(cuts.ravel()).reshape(cuts.shape)
            read = np.where(np.isnan(read), np.inf, read)
            least = np.argmin(read, axis=1)
            better = read[rows, least] < f_best
            best = np.where(better, cuts[rows, least], best)
            f_best = np.where(better, read[rows, least], f_best)
            lower = cuts[rows, np.maximum(least - 1, 0)]
            upper = cuts[rows, np.minimum(least + 1, sections)]
            done = _adjacent(lower, upper) | (upper <= lower) | ((upper - lower) >= (b - a))
            a, b = lower, upper
            if np.all(done):
                break
        return best
    c = b - _INVPHI * (b - a)
    d = a + _INVPHI * (b - a)
    fc = sign * f(c)
    fd = sign * f(d)
    best, f_best = np.where(fc <= fd, c, d), np.minimum(fc, fd)
    for _ in range(_GOLDEN_STEPS):
        # The minimum is in [a, d] where f(c) < f(d), and in [c, b] otherwise.
        # Either way one interior point is kept and one is new.
        lower = fc < fd
        a = np.where(lower, a, c)
        b = np.where(lower, d, b)
        new = np.where(lower, b - _INVPHI * (b - a), a + _INVPHI * (b - a))
        f_new = sign * f(new)
        better = f_new < f_best
        best, f_best = np.where(better, new, best), np.where(better, f_new, f_best)
        c, d = np.where(lower, new, d), np.where(lower, c, new)
        fc, fd = np.where(lower, f_new, fd), np.where(lower, fc, f_new)
        if np.all(_adjacent(a, b) | (b <= a)):
            break
    return best


def _extrema(f: Sampled, t: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Times of the extrema of *f* strictly between grid points.

    For each interior point that is a strict local extremum of the samples,
    the extremum of the function in the two cells beside it. A point with the
    doubles beside it (the neighbourhood of a candidate: where something
    inside the function jumps, or it bends) is one point here, and the cells
    beside it are read on a ladder toward it (:func:`_ladder_turns`) and
    searched where that turns: the function is another piece on each side of
    such a point, and each is asked on its own. The first cell and the last
    the same, toward the end of the window, where no sample lies beyond to
    say whether the function turns in the cell or runs on to its end (a pulse
    that turns 0.01 after the start of the run is above its neighbours at no
    point of the grid).
    """
    if t.size < 3:
        return np.empty(0)
    left, mid, right = u[:-2], u[1:-1], u[2:]
    finite = np.isfinite(left) & np.isfinite(mid) & np.isfinite(right)
    is_max = finite & (mid >= left) & (mid >= right) & ((mid > left) | (mid > right))
    is_min = finite & (mid <= left) & (mid <= right) & ((mid < left) | (mid < right))
    pick = np.nonzero(is_max | is_min)[0]
    # The pairs of points that are doubles next to each other. (No further
    # apart than one spacing of the largest: the test is made of those.)
    close = np.nonzero(np.diff(t) <= float(np.spacing(max(abs(t[0]), abs(t[-1])))))[0]
    close = close[_adjacent(t[close], t[close + 1])]
    tight = np.zeros(t.size - 1, dtype=bool)
    tight[close] = True
    # A sample with a real interval on both sides: its two cells are searched.
    pick = pick[~tight[pick] & ~tight[pick + 1]]
    sign = np.where(is_max[pick], -1.0, 1.0)  # minimize sign * f
    a = t[pick]
    b = t[pick + 2]
    # A neighbourhood, from its first double to its last, is one point with a
    # cell on each side. (It was three points and more, none with a real
    # interval on both sides, and the cells beside it were never searched.
    # `sign(A + 1e-30*sign(B))` has a window of A 1.7e-6 wide whose edge is a
    # few doubles from where B changes sign; with a point on B's change and
    # three doubles on each side of it, all of them a last place outside the
    # window, the window in the next cell had no stop.)
    first = close[np.concatenate(([True], np.diff(close) > 1))] if close.size else close
    last = close[np.concatenate((np.diff(close) > 1, [True]))] + 1 if close.size else close
    inner = (first > 0) & (last < t.size - 1)
    before, after = first[inner] - 1, last[inner] + 1
    lead, trail = u[first[inner]], u[last[inner]]
    with np.errstate(invalid="ignore"):
        up = (lead >= u[before]) & (trail >= u[after]) & ((lead > u[before]) | (trail > u[after]))
        down = (
            (lead <= u[before]) & (trail <= u[after]) & ((lead < u[before]) | (trail < u[after]))
        )
    turns = up | down
    # The stretches read on a ladder: either side of each neighbourhood that
    # is an extremum of the samples, and the two ends of the window. (Each end
    # cell was searched whenever its two samples differed, which they do for
    # anything that is not constant there: eighty steps, a call of the
    # evaluator each, in every scan of every text, 384 calls for the 52 that
    # `abs(time()-120) < 0.5` had taken. A function that runs on to the end
    # of its window has nothing to find there, and its ladder says so in one
    # call.)
    start = last[0] if close.size and first[0] == 0 else 0
    stop = first[-1] if close.size and last[-1] == t.size - 1 else t.size - 1
    # (Where there is a cell beside each end that is not the other's. A window
    # a few thousand doubles long is one neighbourhood from end to end: every
    # point of its grid is the double after the last, `start` is the last
    # point, and the point beside it was read off the end of the grid, which
    # left a leg of a run that short with no stops.)
    ends = np.array([start, stop] if start + 1 < stop else [], dtype=np.intp)
    beside = np.array([start + 1, stop - 1] if start + 1 < stop else [], dtype=np.intp)
    near = np.concatenate((first[inner][turns], last[inner][turns], ends))
    far = np.concatenate((before[turns], after[turns], beside))
    ask = np.isfinite(u[near]) & np.isfinite(u[far])
    # (An end cell that reads the same at both ends is not asked.)
    if ends.size:
        ask[-2:] &= u[ends] != u[beside]
    near, far = near[ask], far[ask]
    more_a, more_b, more_sign = _ladder_turns(f, t[near], u[near], t[far], u[far])
    a = np.concatenate((a, more_a))
    b = np.concatenate((b, more_b))
    sign = np.concatenate((sign, more_sign))
    if a.size == 0:
        return np.empty(0)
    return _least(f, sign, a, b)


def _crossings(
    f: Sampled, nodes: np.ndarray, zero_only: bool, budget: int
) -> tuple[np.ndarray, np.ndarray]:
    """Where *f* crosses a level between consecutive *nodes* (an integer, or
    zero alone), and where it starts or stops being a number.

    A level is crossed on two doubles: the first at which *f* has reached it
    and the first at which it has passed it. They are one double where no
    double gives the level itself, and many apart where *f* is coarser than
    the clock (``time() - 1000`` moves once in thirty doubles of the time). A
    comparison flips on one or the other by whether it is strict, ``floor``
    jumps on the first and ``ceil`` on the second, so both come back.
    """
    u = f(nodes)
    extra = _extrema(f, nodes, u)
    extra = np.unique(extra[(extra > nodes[0]) & (extra < nodes[-1])])
    if extra.size:
        # (Read at the new points alone: the grid was read again for each
        # handful of them.)
        at = np.searchsorted(nodes, extra)
        new = nodes[at] != extra
        if np.any(new):
            nodes = np.insert(nodes, at[new], extra[new])
            u = np.insert(u, at[new], f(extra[new]))
    a, b = u[:-1], u[1:]
    was, now = np.isfinite(a), np.isfinite(b)
    finite = was & now
    numbers = ~np.isnan(a) & ~np.isnan(b)

    edge = np.nonzero(was != now)[0]

    def is_number(times: np.ndarray, live: np.ndarray) -> np.ndarray:
        return np.isfinite(f(times))

    domain = _bisect(is_number, nodes[edge], nodes[edge + 1], was[edge])
    # ...and the last double on the other side of each.
    domain = np.union1d(domain, np.nextafter(domain, -np.inf))

    # Each reading of a level is counted on its own: the two are one change of
    # whatever reads the level.
    found = [np.empty(0)]
    if zero_only:
        for strict in (False, True):

            def side0(values: np.ndarray, strict: bool = strict) -> np.ndarray:
                return values > 0.0 if strict else values >= 0.0

            # An infinite value is on a side of zero like any other: `ln` of
            # the first double past its pole is where a `sign` of it jumps.
            cells = np.nonzero(numbers & (side0(a) != side0(b)))[0]
            if cells.size > budget:
                raise OverBudget(cells.size)

            def side_of_zero(times: np.ndarray, live: np.ndarray, side0=side0) -> np.ndarray:
                return side0(f(times))

            found.append(_bisect(side_of_zero, nodes[cells], nodes[cells + 1], side0(a[cells])))
        return np.unique(np.concatenate(found)), domain

    low = np.where(finite, np.minimum(a, b), 0.0)
    high = np.where(finite, np.maximum(a, b), 0.0)
    for strict in (False, True):
        # Levels reached: low < L <= high. Levels passed: low <= L < high.
        first = np.ceil(low) if strict else np.floor(low) + 1.0
        last = np.ceil(high) - 1.0 if strict else np.floor(high)
        n_levels = np.where(finite, np.maximum(last - first + 1.0, 0.0), 0.0)
        count = float(n_levels.sum())
        if count > budget:
            raise OverBudget(int(min(count, 2**62)))
        cells = np.nonzero(n_levels > 0)[0]
        if cells.size == 0:
            continue
        n = n_levels[cells].astype(np.int64)
        which = np.repeat(cells, n)
        starts = np.cumsum(n) - n
        level = first[which] + (np.arange(which.size) - np.repeat(starts, n))

        def side_of(
            times: np.ndarray, live: np.ndarray, level=level, strict: bool = strict
        ) -> np.ndarray:
            values = f(times)
            return values > level[live] if strict else values >= level[live]

        at_lo = a[which] > level if strict else a[which] >= level
        found.append(_bisect(side_of, nodes[which], nodes[which + 1], at_lo))
    return np.unique(np.concatenate(found)), domain


# Doubles on each side of a point on which an argument is read for how much its
# own rounding moves it, and how many times that it has to move before it is
# taken to be clear of a level it was crossing.
_ROUNDING_DOUBLES = 32
# How many times the moves around a point have to turn round for them to be
# rounding and not the argument's own shape: a wrap turns them twice, a turn
# or a pole once.
_TURNS_OF_ROUNDING = 3
# How many points out from a point, each four times as far as the last, an
# argument is read at for the smallest move it makes: out to 65,536 doubles.
_LEAST_POINTS = 8
# How many points down from a bracket's own length, each a quarter as far out
# as the last, an argument is read at for which side of a level it is on.
_LADDER_POINTS = 12
_CLEAR_OF_ROUNDING = 8.0
# The doublings of one double a bracket grows by where the argument does not
# get clear: sixteen million doubles, a few billionths of the time. An
# argument coarser than the clock, or a staircase on a slow slope, takes that
# long to move by its own rounding. (More where an instant of the run is
# longer than that, and fewer where four instants are shorter: see `_clear`.)
_CLEAR_DOUBLINGS = 24
# ...or an instant of the run where that is longer, and never more than this.
_MOST_DOUBLINGS = 60
# How far the ends of a bracket have to differ, in what the call moves by over
# a stretch of the bracket's own width beside it, for the call to jump inside.
_JUMP_OVER_DRIFT = 4.0


class _Clear(NamedTuple):
    """What :func:`_clear` found around each crossing."""

    below: np.ndarray  # how far before it the argument is clear of its rounding
    above: np.ndarray  # and how far after
    cleared_below: np.ndarray  # whether it got clear there
    cleared_above: np.ndarray
    rounding: np.ndarray  # what the argument's rounding was taken to be
    # It is on both sides of the level by turns for longer than an instant.
    flickers: np.ndarray
    # Not clear on that side as far as a bracket is grown, and not moved at
    # all there: the same on every point it was read at.
    flat_below: np.ndarray
    flat_above: np.ndarray
    # A call inside the argument was read all the way across this crossing's
    # bracket (:func:`_clear_between`), and so is this one.
    read: np.ndarray
    # The crossing is on a jump of something inside the argument.
    on_jump: np.ndarray


def _rounding(f: Sampled, at: np.ndarray) -> tuple:
    """What the argument *f*'s own rounding is taken to be at each of *at*
    (see :func:`_clear`), with what it was read from: the spacing of doubles
    there, the argument there, the stencil's offsets in doubles, the argument
    on the stencil, and at the points 4, 16, 64, ... doubles out."""
    ulp = np.spacing(np.abs(at))
    here = f(at)
    k = np.arange(-_ROUNDING_DOUBLES, _ROUNDING_DOUBLES + 1)
    beside = f((at[:, None] + k[None, :] * ulp[:, None]).ravel()).reshape(at.size, k.size)
    signed = np.diff(beside, axis=1)
    signed[~np.isfinite(signed)] = 0.0
    moves = np.abs(signed)
    # How many times the moves turn round, a move of nothing taking the
    # direction of the one before it.
    way = np.sign(signed)
    last = np.maximum.accumulate(np.where(way != 0.0, np.arange(way.shape[1])[None, :], 0), axis=1)
    way = np.take_along_axis(way, last, axis=1)
    turns = np.count_nonzero(way[:, 1:] * way[:, :-1] < 0.0, axis=1)
    but_one = np.where(turns >= _TURNS_OF_ROUNDING, np.partition(moves, -2, axis=1)[:, -2], 0.0)
    # The moves between points 4, 16, 64, ... doubles out on each side.
    out = ulp[:, None] * (4.0 ** np.arange(1, _LEAST_POINTS + 1))[None, :]
    further = f(
        np.concatenate((at[:, None] - out[:, ::-1], at[:, None] + out), axis=1).ravel()
    ).reshape(at.size, 2 * _LEAST_POINTS)
    wider = np.abs(np.diff(further, axis=1))
    # (Not the one across the point itself, which is the crossing.)
    wider[:, _LEAST_POINTS - 1] = 0.0
    every = np.concatenate((moves, wider), axis=1)
    every[~np.isfinite(every)] = 0.0
    least = np.where(every > 0.0, every, np.inf).min(axis=1)
    rounding = np.maximum(
        np.maximum(but_one, np.where(np.isfinite(least), least, 0.0)),
        np.spacing(np.abs(here)),
    )
    # ...and what it moves by onto the double asked about, from the one
    # before, where that stands alone: several times any other move around
    # it. On a crossing that is a step of the argument's own. (A jump of
    # something inside the argument is known for what it is and is not asked
    # here, `_clear_between`; and beside a pole every move is as large as
    # the next, which is a slope.) A coarse sum beside a term that only
    # shows while the sum is nothing (`L + 1e-20*time()`) moves by 1e-35 on
    # the tread L is zero for and by a whole last place of L onto it;
    # measured by the small moves, the level was clear a few doubles on, and
    # the pulse L makes there some instants later had no stop.
    onto = moves[:, _ROUNDING_DOUBLES - 1]
    next_largest = np.partition(moves, -2, axis=1)[:, -2]
    alone = onto >= _CLEAR_OF_ROUNDING * next_largest
    rounding = np.maximum(rounding, np.where(alone, onto, 0.0))
    return ulp, here, k, beside, further, rounding


def _clear(f: Sampled, at: np.ndarray, zero_only: bool, instant: float) -> _Clear:
    """How far below and above each of *at* the argument *f* has moved clear
    of its own rounding: a power of two of the spacing of doubles there.

    An argument is computed, and near a level it crosses it reads above and
    below the level by turns for as long as it is within its rounding of it.
    That is a few doubles where the argument is steep and a hundred where it
    is turning: ``(1.378*t - 10.953)*t`` moves by its own last bit only every
    fifty doubles of ``t`` near its minimum, and a call of it flickers between
    -22 and -23 over all of them. A bracket has to end outside that stretch on
    both sides for its ends to say what the call does across the crossing.

    The rounding is measured from what *f* moves by from one double to the
    next around the point, and is the largest of these:

    * Where those moves go back and forth, the largest of them but one. An
      argument that jumps there (the wrap of a ``mod`` inside it) makes one
      move the size of the jump, and taken for rounding that put no distance
      clear of it: the bracket ran to its cap and took in every crossing on
      the way. The largest but one, and not a quartile, because rounding comes
      in bursts. In ``frac((time()-55.26)^2/77.056)*4.891 + mod(time(), 47.062)``
      the first term drops by 3.5e-14 once in twelve doubles and the second
      climbs by 4.4e-16 on each, so most moves are a hundredth of the ones
      that carry the argument back over a level; measured by the many, it was
      clear sixteen doubles on, where it had climbed back above the level it
      had just dropped under, and the call read the same at both ends of the
      bracket. Only where they go back and forth, because a slope, a jump and
      a turn do not: beside the pole of ``log(abs(sin(time())))`` every move
      is large and each side of it goes one way, and taken for rounding those
      moves ran the bracket over the whole ladder of levels the argument
      climbs down to the pole.
    * The smallest move it makes at all, here or further out. An argument
      coarser than the clock moves once in many doubles, by one unit of its
      own last place (``time()^2 - 8.302*time() + 17.230801`` near its root: a
      unit of 17.2's), so every move but the crossing's own is zero, and
      measured by them it was clear after a few doubles, inside a stretch of
      thirty thousand where it reads on either side of the level by turns.
      Further out as well as here: a staircase on a slow slope
      (``trunc(time()/0.000101)/2.64 + 1.606*time()`` at t = 100000) makes one
      move in the 64 doubles around a step, the step itself, and its slope
      moves it by a last place only every few thousand doubles. With the step
      taken for the smallest move, the bracket ran to its cap over the level
      the slope crosses next.
    * The spacing of the doubles of its own value.
    * What it moves by onto the crossing itself, where that is several times
      any other move around it.

    *f* is clear where it has moved by :data:`_CLEAR_OF_ROUNDING` times its
    rounding, which a jump is at once. *zero_only* says what its levels are
    (zero alone, or every whole number), for telling an argument that
    flickers across one from an argument that is flat beside a jump; it is
    said to flicker where it changes side of a level time and again over more
    than *instant*.

    The length is a time, and not a number of doubles. An argument coarser
    than the clock is on both sides of a level for a few of its own last
    places, which early in a run are hundreds of doubles of the time:
    `abs(time() - 58.748) + ((time() - 129.929) - 3.935*time())` moves by one
    last place of 72 every hundred and thirty doubles of t = 0.2, and reads
    either side of -72 for five hundred of them, 1.4e-14 of a run to 240.
    """
    ulp, here, k, beside, further, rounding = _rounding(f, at)
    reach, cleared, flats = [], [], []
    # Not clear as far as a bracket is grown, and going back and forth there.
    past_reach = np.zeros(at.size, dtype=bool)
    # How far that is: sixteen million doubles, or an instant where an instant
    # is longer, and no more than four instants.
    #
    # An instant where it is longer: early in a run 2^24 doubles of t are
    # 3.7e-9 of t, less than a billionth of the run wherever t is under a
    # quarter of it. A bracket that ended there said nothing of whether a
    # flicker was longer than an instant, and `sign(time()^2 -
    # 0.790026216393*time() + 0.156035355647)`, on both sides of zero for a
    # sixth of an instant of a run to 1 beside each of its roots, was refused.
    #
    # No more than four: on a clock that is large against the run (a counter
    # started at 2.4e8, a leg late in a long run) 2^24 doubles are thousands
    # of instants, half a time unit, and one bracket took in both edges of a
    # window that long, or two windows, or a plateau that sits on a level:
    # `floor(tanh(40 - 39*(time() - 1e9 - 50)^2)/2 + 0.5)`, one for 1.5 time
    # units, had no stop.
    with np.errstate(over="ignore"):
        instants = np.ceil(np.log2(np.maximum(instant / ulp, 1.0)))
    instants = np.nan_to_num(instants, posinf=_MOST_DOUBLINGS)
    limit = (
        np.minimum(np.maximum(instants, _CLEAR_DOUBLINGS), instants + 2.0)
        if instant > 0.0
        else np.full(at.size, float(_CLEAR_DOUBLINGS))
    )
    limit = np.minimum(limit, _MOST_DOUBLINGS)
    for toward in (-1.0, 1.0):
        step = ulp * toward
        live = np.nonzero(np.isfinite(here))[0]
        ended: list[np.ndarray] = []
        ended_at: list[np.ndarray] = []
        beside_it = np.full(at.size, np.nan)
        flat = np.isfinite(here)
        for doubling in range(int(limit.max()) + 1 if at.size else 0):
            if live.size == 0:
                break
            there = f(at[live] + step[live])
            if doubling == 0:
                beside_it[live] = there
            flat[live] &= there == beside_it[live]
            with np.errstate(invalid="ignore"):
                moved = np.abs(there - here[live]) >= _CLEAR_OF_ROUNDING * rounding[live]
            there, live = there[~moved], live[~moved]
            # As far as theirs is grown.
            capped = doubling >= limit[live]
            ended.append(live[capped])
            ended_at.append(there[capped])
            live = live[~capped]
            step[live] *= 2.0
        live = np.concatenate(ended) if ended else np.empty(0, dtype=int)
        there = np.concatenate(ended_at) if ended_at else np.empty(0)
        got_clear = np.isfinite(here)
        got_clear[live] = False
        if live.size:
            # One that has not moved because it is flat, or a staircase, reads
            # the same wherever its bracket ends. One whose moves turn round
            # time and again is within its rounding of the level all the way,
            # and flickers where it is on both sides of the level by turns
            # for longer than an instant.
            back_and_forth = _flickers(f, at[live], np.abs(step[live]), zero_only, instant)
            # (Of a level it is within reach of. Beside a jump an argument can
            # be flat a quarter of a step from the nearest level, with a last
            # place of rounding on it that goes back and forth:
            # `time() - mod(time() - 46.941, 31.695)` before each wrap.)
            with np.errstate(invalid="ignore"):
                off = np.abs(there) if zero_only else np.abs(there - np.rint(there))
                back_and_forth &= ~(off >= _CLEAR_OF_ROUNDING * rounding[live])
            past_reach[live[back_and_forth]] = True
        reach.append(np.abs(step))
        cleared.append(got_clear)
        flats.append(np.isfinite(here) & ~got_clear & flat)
    # Whether, in a bracket longer than a touch, it is on both sides of a
    # level by turns for longer than an instant: at the points read above, a
    # few doubles to 65,536 from the crossing, and as many again down from
    # the bracket's own length by fours. (Those as well: the crossing a search
    # lands on can be a stray flip at the edge of a flicker six hundred
    # thousand doubles long, with nothing around it for 65,536.)
    #
    # Two turns of the side and not one: a window a few hundred doubles wide,
    # with a clean edge at each end, turns once.
    widest = np.maximum(reach[0], reach[1])
    wide = np.nonzero(np.isfinite(here) & (widest > TOUCH_ULPS * ulp))[0]
    inside = past_reach
    if wide.size:
        # (Down to the 65,536 doubles read above, however far the bracket.)
        rungs = max(
            _LADDER_POINTS, int(np.ceil(0.5 * np.log2((widest[wide] / ulp[wide]).max()))) - 8
        )
        ladder = widest[wide, None] * (0.25 ** np.arange(1, rungs + 1))[None, :]
        ladder = np.concatenate((-ladder, ladder), axis=1)
        near_by = np.concatenate(
            (-(4.0 ** np.arange(_LEAST_POINTS, 0, -1)), k, 4.0 ** np.arange(1, _LEAST_POINTS + 1))
        )
        offsets = np.concatenate((near_by[None, :] * ulp[wide, None], ladder), axis=1)
        values = np.concatenate(
            (
                further[wide, :_LEAST_POINTS],
                beside[wide],
                further[wide, _LEAST_POINTS:],
                f((at[wide, None] + ladder).ravel()).reshape(ladder.shape),
            ),
            axis=1,
        )
        order = np.argsort(offsets, axis=1, kind="stable")
        offsets = np.take_along_axis(offsets, order, axis=1)
        values = np.take_along_axis(values, order, axis=1)
        with np.errstate(invalid="ignore"):
            side = np.sign(values) if zero_only else np.floor(values)
        side[~np.isfinite(values)] = np.inf
        way = np.sign(np.diff(side, axis=1))
        way[~np.isfinite(way)] = 0.0
        last = np.maximum.accumulate(
            np.where(way != 0.0, np.arange(way.shape[1])[None, :], 0), axis=1
        )
        way = np.take_along_axis(way, last, axis=1)
        turns_back = np.count_nonzero(way[:, 1:] * way[:, :-1] < 0.0, axis=1) >= 2
        # From the first point off the side it starts on to the last off the
        # side it ends on.
        rows = np.arange(wide.size)
        left = side != side[:, :1]
        right = side != side[:, -1:]
        begins = np.where(left.any(axis=1), offsets[rows, np.argmax(left, axis=1)], np.inf)
        ends = np.where(
            right.any(axis=1),
            offsets[rows, side.shape[1] - 1 - np.argmax(right[:, ::-1], axis=1)],
            -np.inf,
        )
        long = np.nonzero(
            turns_back & (ends - begins > np.maximum(TOUCH_ULPS * ulp[wide], instant))
        )[0]
        if long.size:
            back = _alternates(f, at[wide[long]], begins[long], ends[long], zero_only)
            inside[wide[long[back]]] = True
    return _Clear(
        reach[0],
        reach[1],
        cleared[0],
        cleared[1],
        rounding,
        inside,
        flats[0],
        flats[1],
        np.zeros(at.size, dtype=bool),
        np.zeros(at.size, dtype=bool),
    )


# Points across a stretch at which an argument is read for whether it
# flickers there, and how many times it has to change side.
_FLICKER_POINTS = 33


def _flickers(
    f: Sampled, at: np.ndarray, reach: np.ndarray, zero_only: bool, instant: float
) -> np.ndarray:
    """Whether *f* goes back and forth over the stretch *reach* on either
    side of each of *at*: read at points across it, its moves from one point
    to the next turn round time and again. That is rounding, around a level
    the argument is within reach of the whole way. A flat stretch does not
    move, a staircase and a slope move one way, and a turn turns once.

    And only where it is also on both sides of a level (zero alone with
    *zero_only*, any whole number without) over more than *instant*, from the
    first of the points off the side it starts on to the last off the side
    it ends on, changing side time and again in between. Beside a turn an
    argument is within its rounding of a level, and its moves are rounding,
    far beyond where it stops crossing the level.
    """
    if at.size == 0:
        return np.zeros(0, dtype=bool)
    across = np.linspace(-1.0, 1.0, _FLICKER_POINTS)
    # Each point a few doubles off its even place, by a different amount. The
    # rounding of a sum of terms repeats every few doubles, and a stretch is a
    # power of two of them long: even points all fell on one place of the
    # pattern, and `1.12e-08 - (time()*time() - 14600*time() + 53290000)`,
    # which reads -1, 0 and 1 units by turns on nine doubles in a row, read as
    # a staircase of one step across sixteen million.
    off = (np.mod(np.arange(across.size) * _INVPHI, 1.0) - 0.5)[None, :] * np.minimum(
        64.0 * np.spacing(np.abs(at)), reach / _FLICKER_POINTS
    )[:, None]
    offsets = across[None, :] * reach[:, None] + off
    values = f((at[:, None] + offsets).ravel()).reshape(at.size, across.size)
    with np.errstate(invalid="ignore"):
        way = np.sign(np.diff(values, axis=1))
    way[~np.isfinite(way)] = 0.0
    last = np.maximum.accumulate(np.where(way != 0.0, np.arange(way.shape[1])[None, :], 0), axis=1)
    way = np.take_along_axis(way, last, axis=1)
    turns = np.count_nonzero(way[:, 1:] * way[:, :-1] < 0.0, axis=1) >= _TURNS_OF_ROUNDING
    with np.errstate(invalid="ignore"):
        side = np.sign(values) if zero_only else np.floor(values)
    side[~np.isfinite(values)] = np.inf
    rows = np.arange(at.size)
    left = side != side[:, :1]
    right = side != side[:, -1:]
    begins = np.where(left.any(axis=1), offsets[rows, np.argmax(left, axis=1)], np.inf)
    ends = np.where(
        right.any(axis=1),
        offsets[rows, side.shape[1] - 1 - np.argmax(right[:, ::-1], axis=1)],
        -np.inf,
    )
    long = np.nonzero(turns & (ends - begins > instant))[0]
    out = np.zeros(at.size, dtype=bool)
    if long.size:
        out[long[_alternates(f, at[long], begins[long], ends[long], zero_only)]] = True
    return out


# How many times an argument has to change side, from one to the next of the
# points it is read at across the middle of a stretch, to be flickering there.
_SIDE_CHANGES = 4


def _alternates(
    f: Sampled, at: np.ndarray, begins: np.ndarray, ends: np.ndarray, zero_only: bool
) -> np.ndarray:
    """Whether *f* goes from one side of a level to the other time and again
    at points across the middle of the stretch from ``at + begins`` to
    ``at + ends``: the stretch it is off the side it starts on at one end of,
    and off the side it ends on at the other.

    That is what tells a flicker from a window. A window with noise at its
    edges is off its side from one edge to the other as well, and turns round
    more than once in the noise: `sign((2.4e-07 - L)*(2.4e-07 + L))` for a
    coarse sum L is zero, one and zero again over 2.8e-07, an instant and
    more of a run to 240, and read by its ends alone it was refused as a
    flicker in that run and had its stops in a shorter one. Between its edges
    a window is on one side, or on a few in turn; a flicker is on each of two
    at every other point.
    """
    across = np.linspace(0.1, 0.9, _FLICKER_POINTS)
    # (Each point off its even place by a different amount, as in `_flickers`.)
    width = ends - begins
    off = (np.mod(np.arange(across.size) * _INVPHI, 1.0) - 0.5)[None, :] * np.minimum(
        64.0 * np.spacing(np.abs(at)), width / (4.0 * _FLICKER_POINTS)
    )[:, None]
    values = f(
        (at[:, None] + begins[:, None] + across[None, :] * width[:, None] + off).ravel()
    ).reshape(at.size, across.size)
    with np.errstate(invalid="ignore"):
        side = np.sign(values) if zero_only else np.floor(values)
    side[~np.isfinite(values)] = np.inf
    return np.count_nonzero(side[:, 1:] != side[:, :-1], axis=1) >= _SIDE_CHANGES


def _clear_between(
    f: Sampled,
    near: np.ndarray,
    zero_only: bool,
    instant: float,
    zones: tuple | None,
    inside: Sequence[_Held] = (),
    knots: np.ndarray | None = None,
) -> _Clear:
    """:func:`_clear` for the crossings *near* of an argument that are not
    where something inside it jumps, and for one that is, the stretch a call
    inside was read over for its jump (*zones*: where each begins and ends,
    sorted and apart, and whether the call was read all the way across it),
    or a touch each way where no such stretch holds it: on a double where a
    call or a comparison inside changes (*inside*: what each holds from one
    jump to the next; a pass can find a jump of the call around that it did
    not find for the call inside), or on a breakpoint of a table (*knots*,
    sorted).

    The argument of ``mod(floor(time()), 24)`` is a staircase: it jumps by a
    twenty-fourth where ``floor`` jumps, onto a whole number once a day, and
    is flat for a time unit on each side. Nothing else of it moves, so its
    rounding was taken to be that jump, and it was never clear of the level
    it had jumped onto. But where the calls inside an argument jump is known,
    each of them read for its own jumps before this one, and what the
    argument does there is their jump and no rounding of its own: this call
    changes where they do or not at all. Where a call inside was read across
    a stretch, for an argument of its own that is never clear there, this one
    is read across the same stretch. A comparison standing as a number
    (``(time() > 10) + (time() > 20)``) and a table that steps are staircases
    of the same kind.
    """
    if near.size == 0 or (zones is None and not inside and (knots is None or not knots.size)):
        return _clear(f, near, zero_only, instant)
    touch = TOUCH_ULPS * np.spacing(np.abs(near))
    below, above = touch.copy(), touch.copy()
    read = np.zeros(near.size, dtype=bool)
    on = np.zeros(near.size, dtype=bool)
    if zones is not None:
        begins, ends, across = zones
        which = np.searchsorted(begins, near, side="right") - 1
        on = (which >= 0) & (near <= ends[np.maximum(which, 0)])
        which = which[on]
        ulp = np.spacing(np.abs(near[on]))
        below[on] = np.maximum(near[on] - begins[which], ulp)
        above[on] = np.maximum(ends[which] - near[on], ulp)
        read[on] = across[which]
    off = np.nonzero(~on)[0]
    if off.size and knots is not None and knots.size:
        after = np.minimum(np.searchsorted(knots, near[off]), knots.size - 1)
        gap = np.minimum(
            np.abs(knots[after] - near[off]), np.abs(near[off] - knots[np.maximum(after - 1, 0)])
        )
        on[off[gap <= touch[off]]] = True
        off = np.nonzero(~on)[0]
    if off.size and inside:
        before = np.nextafter(near[off], -np.inf)
        jumps = np.zeros(off.size, dtype=bool)
        for held in inside:
            jumps |= _differs(held.reading(near[off]), held.reading(before), held.by)
        on[off[jumps]] = True
    if not on.any():
        return _clear(f, near, zero_only, instant)
    rest = _clear(f, near[~on], zero_only, instant)

    def over(of_rest: np.ndarray, of_these) -> np.ndarray:
        out = np.empty(near.size, dtype=of_rest.dtype)
        out[~on] = of_rest
        out[on] = of_these
        return out

    return _Clear(
        over(rest.below, below[on]),
        over(rest.above, above[on]),
        over(rest.cleared_below, True),
        over(rest.cleared_above, True),
        over(rest.rounding, np.spacing(np.abs(np.nan_to_num(f(near[on]))))),
        over(rest.flickers, False),
        over(rest.flat_below, False),
        over(rest.flat_above, False),
        over(rest.read, read[on]),
        over(rest.on_jump, True),
    )


def _zones(entries: list) -> tuple | None:
    """The stretches calls were read over for their jumps (each entry: where
    they begin, where they end, whether the call was read all the way
    across), sorted, and those that overlap as one."""
    entries = [entry for entry in entries if entry[0].size]
    if not entries:
        return None
    begins = np.concatenate([entry[0] for entry in entries])
    ends = np.concatenate([entry[1] for entry in entries])
    read = np.concatenate([np.full(entry[0].size, entry[2], dtype=bool) for entry in entries])
    order = np.argsort(begins, kind="stable")
    begins, ends, read = begins[order], ends[order], read[order]
    reach = np.maximum.accumulate(ends)
    starts = np.nonzero(np.concatenate(([True], begins[1:] > reach[:-1])))[0]
    return (
        begins[starts],
        np.maximum.reduceat(ends, starts),
        np.logical_or.reduceat(read, starts),
    )


class _Brackets(NamedTuple):
    """One bracket for each run of crossings whose brackets overlap."""

    lo: np.ndarray
    hi: np.ndarray
    lo_clear: np.ndarray  # whether the argument got clear at that end
    hi_clear: np.ndarray
    rounding: np.ndarray  # the largest of the crossings' in it
    flickers: np.ndarray  # the argument is on both sides of a level by turns inside
    # An end that is a cut: another crossing is within the bracket's own width
    # beyond it.
    lo_cut: np.ndarray
    hi_cut: np.ndarray
    # What is inside is one crossing, the few doubles a level is crossed on,
    # and the argument is clear of its rounding at each end, or flat up to
    # one end and clear at the other.
    plain: np.ndarray
    # An end the argument is not clear at and has not moved at all up to.
    lo_flat: np.ndarray
    hi_flat: np.ndarray
    # Every crossing inside is on a jump of something inside the argument (a
    # step call, a comparison, a table's breakpoint): the call changes on
    # those doubles or not at all.
    jump: np.ndarray


def _brackets(
    f: Sampled,
    near: np.ndarray,
    zero_only: bool,
    instant: float,
    zones: tuple | None = None,
    inside: Sequence[_Held] = (),
    knots: np.ndarray | None = None,
    never_back: bool = False,
) -> _Brackets:
    """The brackets around the crossings *near* (sorted) of the argument *f*,
    whose levels are the whole numbers, or zero alone. *zones* are where the
    step calls inside the argument were read for their jumps
    (:func:`_zones`), *inside* what each of those calls, and each comparison
    in the argument, holds from one jump to the next, and *knots* the
    breakpoints of the tables it reads (sorted). *never_back*: the argument
    never steps back (:func:`_a_line_in_time`), and a bracket with one
    crossing in it is plain whether it is clear at its ends or not."""
    clear = _clear_between(f, near, zero_only, instant, zones, inside, knots)
    lo = near - clear.below
    hi = near + clear.above
    cleared_below, cleared_above = clear.cleared_below.copy(), clear.cleared_above.copy()
    flat_below, flat_above = clear.flat_below.copy(), clear.flat_above.copy()
    # Two crossings whose brackets run into each other, with the argument
    # clear of every level half way between them: two brackets that end
    # there. How far an argument has to move to be clear is measured by its
    # rounding at the crossing, and that can be many times its rounding a
    # little further on, so the bracket of each edge of a window can run over
    # its top and out the other side, where it reads what it read where it
    # began.
    reach = np.maximum.accumulate(hi)
    joined = np.nonzero(
        (lo[1:] <= reach[:-1]) & (near[1:] - near[:-1] > TOUCH_ULPS * np.spacing(np.abs(near[1:])))
    )[0]
    cut_at = np.empty(0)
    if joined.size:
        between = 0.5 * (near[joined] + near[joined + 1])
        value = f(between)
        off = np.abs(value) if zero_only else np.abs(value - np.rint(value))
        with np.errstate(invalid="ignore"):
            apart = np.isfinite(value) & (off >= _CLEAR_OF_ROUNDING * _rounding(f, between)[-1])
        cut, cut_at = joined[apart], between[apart]
        if cut.size:
            # Each crossing's bracket ends at the first cut after it and
            # begins after the last cut before it. (For all of them at once:
            # cut by cut, a staircase of 300,000 noisy steps, each bracket run
            # into the next, took hours to be refused.)
            n_before = np.searchsorted(cut, np.arange(near.size), side="left")
            after = np.where(
                n_before < cut.size, cut_at[np.minimum(n_before, cut.size - 1)], np.inf
            )
            short = hi > after
            hi[short] = after[short]
            cleared_above[short] = True
            flat_above[short] = False
            # (From the next double on: brackets that touch are one.)
            before = np.where(
                n_before > 0, np.nextafter(cut_at[np.maximum(n_before - 1, 0)], np.inf), -np.inf
            )
            short = lo < before
            lo[short] = before[short]
            cleared_below[short] = True
            flat_below[short] = False
    # One bracket for each run that overlap, taken in the order of where they
    # begin. (Not in the order of the crossings. How far a bracket reaches is
    # a power of two of doubles, and of two crossings the later can begin the
    # earlier: taken in the crossings' order, the brackets of two steps of a
    # staircase came out as two that overlapped by a sixth of a tread, and a
    # condition on it had one stop more on every other grid.)
    order = np.argsort(lo, kind="stable")
    lo, hi = lo[order], hi[order]
    cleared_below, cleared_above = cleared_below[order], cleared_above[order]
    flat_below, flat_above = flat_below[order], flat_above[order]
    reach = np.maximum.accumulate(hi)
    first = np.concatenate(([True], lo[1:] > reach[:-1]))
    starts = np.nonzero(first)[0]
    # An end is where the argument got clear only if the crossing that put it
    # there did.
    lo_clear = np.minimum.reduceat(np.where(cleared_below, lo, np.inf), starts)
    hi_clear = np.maximum.reduceat(np.where(cleared_above, hi, -np.inf), starts)
    lo_flat = np.minimum.reduceat(np.where(flat_below, lo, np.inf), starts)
    hi_flat = np.maximum.reduceat(np.where(flat_above, hi, -np.inf), starts)
    lo = np.minimum.reduceat(lo, starts)
    hi = np.maximum.reduceat(hi, starts)
    lo_clear, hi_clear = lo_clear == lo, hi_clear == hi
    lo_flat, hi_flat = lo_flat == lo, hi_flat == hi
    rounding = np.maximum.reduceat(clear.rounding[order], starts)
    flickers = np.logical_or.reduceat(clear.flickers[order], starts)
    lo_cut = np.isin(lo, np.nextafter(cut_at, np.inf))
    hi_cut = np.isin(hi, cut_at)
    # A bracket is plain where the crossings in it are the few doubles one
    # level is crossed on and the argument got clear of its rounding at both
    # ends. What the call reads at its two ends then says whether it jumps in
    # between, and one bisection finds where.
    #
    # Clear at one end and flat up to the other is plain too. An argument
    # that jumps onto a level and sits there (`floor(time())/24` on a whole
    # number, in `mod(floor(time()), 24)`), or leaves one it was held on
    # (`max(0, time() - 5)`), is clear at once on the side it moves on and
    # never on the other, where nothing moves.
    #
    # Of any other bracket the ends say nothing. One the argument is not
    # clear at the end of ends inside whatever the argument is doing there,
    # and one with crossings more than a touch apart holds more than one
    # change: the two edges of a window a few roundings deep, a pulse a few
    # instants before a step, a train of them. Each rule that read such a
    # bracket by its ends lost something with no word (see `_combed`), so it
    # is not read by its ends.
    which = np.searchsorted(lo, near, side="right") - 1
    first_of = np.minimum(np.searchsorted(which, np.arange(lo.size), side="left"), near.size - 1)
    last_of = np.maximum(np.searchsorted(which, np.arange(lo.size), side="right") - 1, 0)
    one = near[last_of] - near[first_of] <= TOUCH_ULPS * np.spacing(np.abs(near[last_of]))
    # (Read across wherever a call inside was.)
    across = np.logical_or.reduceat(clear.read[order], starts)
    return _Brackets(
        lo,
        hi,
        lo_clear,
        hi_clear,
        rounding,
        flickers,
        lo_cut,
        hi_cut,
        (((lo_clear | (lo_flat & hi_clear)) & (hi_clear | (hi_flat & lo_clear))) | never_back)
        & one
        & ~across,
        lo_flat,
        hi_flat,
        np.logical_and.reduceat(clear.on_jump[order], starts) & ~across,
    )


# Calls whose value is a sawtooth: it climbs with its argument and drops back.
# The rest of the step calls hold one value from a jump to the next.
_SAWTOOTH = ("frac", "mod")


class _Held(NamedTuple):
    """What of a call stays as it is from one of its jumps to the next."""

    reading: Sampled
    by: float  # how far two readings of it may differ and be the same
    # For a sawtooth, the call's own value in units of what it drops by where
    # it wraps; ``None`` for a call that steps, whose reading is its value.
    wraps: Sampled | None


def _held(ev: Evaluate, call: str) -> _Held | None:
    """What of *call* stays as it is from one of its jumps to the next: the
    value of a call that steps, and for a sawtooth the whole number it takes
    off its argument (``x - frac(x)``, and ``(x - mod(x, m))/m`` for a modulus
    the clock does not move). ``None`` for a call that is neither.
    """
    m = _CALL_NAME.match(call)
    if m is None or m.group(1) not in _PHASES or _close_paren(call, m.end() - 1) != len(call) - 1:
        return None
    name = m.group(1)
    if name not in _SAWTOOTH:
        return _Held(functools.partial(ev, call), 0.0, None)
    args = [a.strip() for a in _split_commas(call[m.end() : -1])]
    if name == "frac" and len(args) == 1:
        return _Held(
            functools.partial(ev, f"(({args[0]})-({call}))"), 0.5, functools.partial(ev, call)
        )
    if name == "mod" and len(args) == 2 and not _READS_CLOCK.search(args[1]):
        return _Held(
            functools.partial(ev, f"((({args[0]})-({call}))/({args[1]}))"),
            0.5,
            functools.partial(ev, f"(({call})/({args[1]}))"),
        )
    return None


def _differs(a: np.ndarray, b: np.ndarray, by: float) -> np.ndarray:
    """Whether two readings differ by more than *by*, not a number being a
    reading of its own."""
    with np.errstate(invalid="ignore"):
        return (np.abs(a - b) > by) | (np.isnan(a) != np.isnan(b))


def _call_jumps(
    ev: Evaluate,
    call: str,
    f: Sampled,
    near: np.ndarray,
    span: tuple[float, float] | None = None,
    zero_only: bool = False,
    instant: float = 0.0,
    *,
    brackets: _Brackets | None = None,
    within: tuple[float, float] | None = None,
    zones: list | None = None,
    line: bool = False,
) -> np.ndarray:
    """The first double of the call's new value around each crossing of its
    argument in *near*, where the call jumps across it.

    The bracket around a crossing ends where the argument is clear of its own
    rounding on each side (:func:`_clear`), and brackets that overlap are one:
    the two doubles a level is crossed on are one crossing, and so is a level
    the argument flickers across for no longer than *instant*.

    Where it flickers across one for longer, the call is refused
    (:class:`Flickers`). A seventh power written out term by term is
    within its rounding of zero for a hundredth of the time on each side of
    its root, and ``sign`` of it reads -1 and 1 by turns over all of that.
    There is no time at which such a call changes: a stop on a double where
    it reads its new value restarts the run where it reads the old one again
    a double later, a stop where the flicker ends is past whatever else
    closes a window before that, and what the rest of a text does inside the
    flicker cannot be told from the flicker. Each of those was tried, and
    each gave a run that returned nothing for a window it had stepped over.

    Across a plain bracket (one crossing inside, and the argument clear at
    its ends) the call jumps where it holds another reading at one end than
    at the other (:func:`_across`): its value for a call that steps, and for
    a sawtooth (``mod``, ``frac``) the whole number it takes off its
    argument, which is the same at both ends where the argument only touches
    a level. (A call nothing is known to hold of is read by how far its two
    ends differ beside what it drifts by: :func:`_jumps_across`.)

    A crossing on a jump of something inside the argument (a step call, a
    comparison, a table's breakpoint) is asked on that double. Any other
    bracket is read from one end to the other (:func:`_combed`).

    *zones*, a list, is given where the call was read for its jumps: each
    stretch it was read over, and whether all the way across
    (:func:`_zones`).

    With *span*, the window the crossings are *call*'s own argument's in (and
    *zero_only* as for its phase), what the brackets found is then held
    against the call itself (:func:`_missed_between`). For a call standing as
    a number only.

    With *line*, the argument is a line in time (:func:`_a_line_in_time`),
    whose brackets are plain whether it is clear at their ends or not. What
    the call holds is then read from the end of each bracket to the beginning
    of the next as well, and from each end of *within* to the bracket beside
    it. A line never steps back, so a reading anywhere on it is the call's
    there; and a phase is not always the arithmetic its call does. ``mod`` is
    the remainder exactly and its phase the quotient rounded: where two
    values of a coarse x in a row give a quotient that rounds to k with
    ``k*m`` between them, the wrap is on the step from one to the other,
    after the bracket of the double the phase reached k on and before the
    bracket of the one it passed it on. (Read by its brackets alone,
    ``mod(1e12 + time(), 0.0017)`` over (0, 1.7) had 330 stops for its 1,000
    wraps; left out of the rule and read every quarter instant instead, it
    was refused as more than can be read at offsets from 1e8 up, where the
    revisions before had every wrap.)
    """
    if within is None:
        within = span
    if near.size == 0:
        # (A window a few doubles of a coarse line long can hold a wrap of
        # `mod` of it and no crossing of the phase at all.)
        if not line or within is None:
            return near
        held = _held(ev, call)
        if held is None:
            return near
        return _in_the_gaps(held, np.array([within[0]]), np.array([within[1]]), zones)
    near = np.sort(near)
    found_at = _brackets(f, near, zero_only, instant) if brackets is None else brackets
    lo, hi = found_at.lo, found_at.hi
    if found_at.flickers.any():
        # (The first of them. A long flicker is many crossings, each with a
        # bracket of its own, and a call can flicker at more than one place:
        # from the first to the last named a stretch with clean time in it.)
        first = int(np.argmax(found_at.flickers))
        raise Flickers(call, lo[first], hi[first], int(found_at.flickers.sum()))
    held = _held(ev, call)
    jump = found_at.jump
    if held is None:
        # (Nothing to tell a jump of this text by: what it is at the two ends.)
        jump = np.zeros(lo.size, dtype=bool)
    plain = (found_at.plain | found_at.jump) & ~jump
    found = (
        _jumps_across(
            ev, call, lo[plain], hi[plain], found_at.lo_cut[plain], found_at.hi_cut[plain]
        )
        if held is None
        else _across(held, lo[plain], hi[plain])
    )
    if jump.any():
        # Where something inside the argument jumps, the call changes on that
        # double or not at all. (Not read by the two ends of the few doubles
        # around it: `frac(floor(time()/0.001) + 0.3*time())` does not move
        # where `floor` jumps, and is 1.5e-11 of rounding on every double,
        # which across a bracket that short was more than it drifts.)
        on = near[jump[np.searchsorted(lo, near, side="right") - 1]]
        found = np.union1d(found, on[_jumps_on(held, on)])
    if zones is not None:
        zones.append((lo[plain | jump], hi[plain | jump], False))
    if line and held is not None:
        gap_lo, gap_hi = hi[:-1], lo[1:]
        if within is not None:
            gap_lo = np.concatenate(([within[0]], hi))
            gap_hi = np.concatenate((lo, [within[1]]))
        found = np.union1d(found, _in_the_gaps(held, gap_lo, gap_hi, zones))
    if not (plain | jump).all():
        rest = ~(plain | jump)
        if held is None:
            # Nothing of this call stays as it is between two jumps (a `mod`
            # whose modulus moves with the clock), so there is nothing to
            # read it by where its ends do not say what it does.
            first = int(np.argmax(rest))
            raise Flickers(call, lo[first], hi[first], int(rest.sum()))
        # ...as far out as the argument is within its rounding of a level,
        # which can be further than the bracket was grown.
        wide_lo, wide_hi = _to_clear(f, found_at, rest, zero_only, within)
        if instant > 0.0:
            cells = np.ceil((wide_hi - wide_lo) / (_COMB_OF_AN_INSTANT * instant))
            if float(cells.sum()) > _COMB_BUDGET:
                longest = int(np.argmax(wide_hi - wide_lo))
                raise Flickers(
                    call, wide_lo[longest], wide_hi[longest], int(wide_lo.size), unread=True
                )
        if zones is not None:
            zones.append((wide_lo, wide_hi, True))
        inside = _combed(held, near, wide_lo, wide_hi, instant)
        # (A change of what a sawtooth takes off its argument is a jump of
        # the sawtooth where it moves there.)
        found = np.union1d(found, inside[_moves(held, inside)])
    if held is None or span is None:
        return found
    more = _missed_between(
        f,
        held,
        found,
        lo,
        hi,
        found_at.lo_clear,
        found_at.hi_clear,
        found_at.rounding,
        span,
        zero_only,
    )
    if more.size:
        if zones is not None:
            touch = TOUCH_ULPS * np.spacing(np.abs(more))
            zones.append((more - touch, more + touch, False))
        # A jump no bracket holds is one the argument flickers past them for.
        around = _clear(f, more, zero_only, instant)
        if around.flickers.any():
            first = int(np.argmax(around.flickers))
            raise Flickers(
                call,
                (more - around.below)[first],
                (more + around.above)[first],
                int(around.flickers.sum()),
            )
    return np.union1d(found, more) if more.size else found


# How many changes are looked for in one gap between two brackets of a line:
# a wrap that comes after the bracket of the level before, and the next one
# before the bracket of its own.
_GAP_CHANGES = 3


def _in_the_gaps(held: _Held, lo: np.ndarray, hi: np.ndarray, zones: list | None) -> np.ndarray:
    """The changes of what a call of a line holds from each *lo* to its *hi*,
    the stretches between the brackets of its argument's crossings: the first
    double of each new reading, where the call moves there.

    One change is bisected for at a time, and the rest of the gap is read
    again from it: the rounded quotient ``mod`` is read by can pass one level
    a double of a coarse argument early and reach the next one a double late,
    and both wraps are then in the gap between those two brackets. (No more
    than a few are looked for. A call with another argument that is no line
    can change any number of times in a gap, each at a crossing of that
    argument's, which has its own bracket.)
    """
    reading, by, _ = held
    open_ = hi > lo
    lo, hi = lo[open_], hi[open_]
    every = []
    for _ in range(_GAP_CHANGES):
        if lo.size == 0:
            break
        was, then = reading(lo), reading(hi)
        changed = np.nonzero(_differs(was, then, by) & np.isfinite(was) & np.isfinite(then))[0]
        if changed.size == 0:
            break
        was = was[changed]

        def as_it_was(times: np.ndarray, live: np.ndarray, was=was) -> np.ndarray:
            return ~_differs(reading(times), was[live], by)

        at = _bisect(as_it_was, lo[changed], hi[changed], np.ones(changed.size, dtype=bool))
        every.append(at)
        lo, hi = at, hi[changed]
        open_ = hi > lo
        lo, hi = lo[open_], hi[open_]
    if not every:
        return np.empty(0)
    found = np.unique(np.concatenate(every))
    found = found[_moves(held, found)]
    if zones is not None and found.size:
        touch = TOUCH_ULPS * np.spacing(np.abs(found))
        zones.append((found - touch, found + touch, False))
    return found


def _across(held: _Held, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The first double of a call's new reading inside each plain bracket
    ``[lo, hi]`` it holds another reading at the end of than at the beginning
    (*held*: what it keeps from one jump to the next).

    The argument is clear of its rounding at both ends of a plain bracket (or
    held flat up to one), so what the call holds there is what it holds on
    that side of the crossing, and the two differ where it jumps across. A
    sawtooth was read by its value instead, by how far the two ends differ
    beside how far it drifts over a bracket's width on either side: where
    ``1/(1 + exp(...))`` underflows to nothing and ``frac`` of it with it,
    that was a jump on one grid and none on the next, by where the bracket
    happened to end.
    """
    if lo.size == 0:
        return np.empty(0)
    reading, by, _ = held
    was, then = reading(lo), reading(hi)
    changed = np.nonzero(_differs(was, then, by) & np.isfinite(was) & np.isfinite(then))[0]
    if changed.size == 0:
        return np.empty(0)
    was = was[changed]

    def as_it_was(times: np.ndarray, live: np.ndarray) -> np.ndarray:
        return ~_differs(reading(times), was[live], by)

    found = np.unique(
        _bisect(as_it_was, lo[changed], hi[changed], np.ones(changed.size, dtype=bool))
    )
    return found[_moves(held, found)]


# The cells a bracket is read on at the least, how long one is at the most
# (in instants), and how many points are evaluated at a time.
_COMB_CELLS = 16
_COMB_OF_AN_INSTANT = 0.25
_COMB_POINTS = 1 << 20
# The cells a call is read on in all, over every bracket of one pass: as many
# as the finest grid a pass is made on has. A call whose argument is within
# its rounding of a level for more of the run than that is refused.
_COMB_BUDGET = 1 << 21


def _to_clear(
    f: Sampled,
    found_at: _Brackets,
    which: np.ndarray,
    zero_only: bool,
    within: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """The brackets *which* of *found_at* with each end the argument *f* is
    not clear at moved out to where it is clear of every level by its
    rounding (:data:`_CLEAR_OF_ROUNDING` times over), to where it stops being
    a number, or to the end of the window *within*; those that then overlap
    as one.

    A bracket is grown a few instants each way (:func:`_clear`). An argument
    that is a staircase of its own last places can be within one of them of a
    level for thousands of instants more, and step back across it out there:
    ``(2.25*time() + 1e9) - 0.9*time()`` goes up by one last place, back and
    up again at one step in six, each stretch a few instants to thousands
    long by the length of the run. Read as far as the bracket alone, a step
    had one stop and the pulse beside it none, in 26 of 60 windows drawn.

    An end the argument has not moved at all up to is held where it is for
    as long as it does not move: it goes no further out than the other end
    has to, and nowhere where the other end is clear, or as flat. (A tread of
    a staircase is flat for as far as a bracket is grown as well, and moves
    further out; an argument held on a level, as ``tanh`` of a large number
    is on one, does not, and all of a plateau is not read for its edge.)
    """
    lo, hi = found_at.lo[which], found_at.hi[which]
    rounding = found_at.rounding[which]
    first, last = (-np.inf, np.inf) if within is None else within
    clear = (found_at.lo_clear[which], found_at.hi_clear[which])
    flat = [found_at.lo_flat[which] & ~clear[0], found_at.hi_flat[which] & ~clear[1]]
    ends = [lo.copy(), hi.copy()]
    done = [clear[0] | (lo <= first), clear[1] | (hi >= last)]
    was = (f(lo), f(hi))
    step = np.maximum(hi - lo, np.spacing(np.maximum(np.abs(lo), np.abs(hi))))
    for _ in range(_MOST_DOUBLINGS + 4):
        # One that is flat so far stops where the other has.
        for side in (0, 1):
            done[side] = done[side] | (flat[side] & done[1 - side])
        if done[0].all() and done[1].all():
            break
        for side, toward, bound in ((0, -1.0, first), (1, 1.0, last)):
            live = np.nonzero(~done[side])[0]
            if live.size == 0:
                continue
            there = (lo, hi)[side][live] + toward * step[live]
            there = np.maximum(there, bound) if toward < 0.0 else np.minimum(there, bound)
            value = f(there)
            with np.errstate(invalid="ignore"):
                off = np.abs(value) if zero_only else np.abs(value - np.rint(value))
                far = ~np.isfinite(value) | (off >= _CLEAR_OF_ROUNDING * rounding[live])
            flat[side][live] &= value == was[side][live]
            ends[side][live] = there
            done[side][live[far | (there == bound)]] = True
        step = 2.0 * step
    # An end that never moved stays where the bracket put it unless the other
    # end moved, and was read further out for it.
    moved = [~flat[side] & (ends[side] != (lo, hi)[side]) for side in (0, 1)]
    lo = np.where(flat[0] & ~moved[1], lo, ends[0])
    hi = np.where(flat[1] & ~moved[0], hi, ends[1])
    order = np.argsort(lo, kind="stable")
    lo, hi = lo[order], hi[order]
    reach = np.maximum.accumulate(hi)
    starts = np.nonzero(np.concatenate(([True], lo[1:] > reach[:-1])))[0]
    return lo[starts], np.maximum.reduceat(hi, starts)


def _combed(
    held: _Held, near: np.ndarray, lo: np.ndarray, hi: np.ndarray, instant: float
) -> np.ndarray:
    """The first double of each new reading of a call inside the brackets
    ``[lo, hi]`` (sorted, apart), which hold the crossings *near* of its
    argument: what the call keeps from one jump to the next (*held*) is read
    every quarter of an *instant* from one end of a bracket to the other, and
    each change from one point to the next is bisected.

    For a bracket that is not plain: the argument is not clear of its
    rounding at an end of it, or crosses more than once inside. An argument
    that is a small sum on a large one (``1e9 + 3e-7*sin(...)``) moves by a
    few of its own last places in all, is never clear of a level, and is no
    flicker either: it is a staircase, and a call of it has clean jumps. What
    the call reads at the two ends of such a bracket does not say what it
    does in between, and each way of reading it from them lost something
    with no word. One bisection between two ends that differ found the first
    change it met, a pulse a few instants before a step, and the step had no
    stop. Ends that read the same were no jump, across a window a few
    roundings deep. A bracket cut in two where the argument was on a third
    side had one jump in each half, and a train of pulses had one stop for
    372 changes. A sawtooth's ends were held against what it does a
    bracket's width further out, which was the other edge of its window.

    So the call is read. A stretch it holds one reading for is found where
    it is a quarter of an instant long, wherever it is in the bracket and
    however many there are, and a shorter one where it is a sixteenth of the
    bracket, or begins or ends on a crossing found for the argument (the call
    is read a touch on each side of every crossing, and half way from each
    to the next). What is shorter than that is within an instant of a stop
    or is a touch.

    A bracket is no longer than a few instants each way of each crossing in
    it (:func:`_clear`) and as far beyond as its argument is not clear
    (:func:`_to_clear`): a few dozen evaluations for most.
    """
    if lo.size == 0:
        return np.empty(0)
    cells = np.full(lo.size, float(_COMB_CELLS))
    if instant > 0.0:
        cells = np.maximum(cells, np.ceil((hi - lo) / (_COMB_OF_AN_INSTANT * instant)))
    n = np.minimum(cells, 2.0**40).astype(np.int64)
    # The crossings in each bracket: a touch on each side of each, and half
    # way from each to the next where that is more than a touch.
    owner = np.searchsorted(lo, near, side="right") - 1
    held_in = (owner >= 0) & (near <= hi[np.maximum(owner, 0)])
    at, owner = near[held_in], owner[held_in]
    touch = TOUCH_ULPS * np.spacing(np.abs(at))
    apart = (owner[1:] == owner[:-1]) & (at[1:] - at[:-1] > touch[1:])
    beside = np.concatenate((at - touch, at + touch, 0.5 * (at[:-1] + at[1:])[apart]))
    whose = np.concatenate((owner, owner, owner[1:][apart]))
    beside = np.clip(beside, lo[whose], hi[whose])

    reading, by, _ = held
    found: list[np.ndarray] = []
    # (A million points at a time where the brackets are many or long: a
    # staircase of a hundred thousand steps, none of them clear, is millions.)
    ends = np.cumsum(n + 1)
    starts = ends - (n + 1)
    total = int(ends[-1])
    begin = 0
    while True:
        end = min(begin + _COMB_POINTS, total)
        g = np.arange(begin, end)
        which = np.searchsorted(ends, g, side="right")
        j = g - starts[which]
        even = np.where(j == n[which], hi[which], lo[which] + (hi - lo)[which] * (j / n[which]))
        points = np.unique(
            np.concatenate((even, beside[(beside >= even[0]) & (beside <= even[-1])]))
        )
        bracket = np.searchsorted(lo, points, side="right") - 1
        read = reading(points)
        changed = np.nonzero(_differs(read[:-1], read[1:], by) & (bracket[:-1] == bracket[1:]))[0]
        if changed.size:
            was = read[changed]

            def as_it_was(times: np.ndarray, live: np.ndarray, was=was) -> np.ndarray:
                return ~_differs(reading(times), was[live], by)

            found.append(
                _bisect(
                    as_it_was,
                    points[changed],
                    points[changed + 1],
                    np.ones(changed.size, dtype=bool),
                )
            )
        if end >= total:
            break
        # (From the last point of this lot, which the next begins on.)
        begin = end - 1
    return np.unique(np.concatenate(found)) if found else np.empty(0)


# Doubles from a jump at which a sawtooth is read for how far one double of
# the clock moves it: on each side of the jump, and further out.
_DRIFT_AT = (-2.0, 1.0, -64.0, 64.0, -4096.0, 4096.0)
# The least a sawtooth's value moves by where it wraps, in wraps, however
# coarse its argument: the rounding it is held against is no more than this.
_WRAP_AT_LEAST = 0.25


def _moves(held: _Held, at: np.ndarray) -> np.ndarray:
    """Whether a call jumps on each of the doubles *at*, on which what it
    holds from one jump to the next (*held*) changes.

    A call that steps does: what it holds is its value. What a sawtooth
    takes off its argument also changes where the argument itself jumps by a
    whole number, and the sawtooth does not move there:
    ``frac(floor(time()) + 0.3*time())`` is ``frac(0.3*time())``, and had a
    stop on every whole number of the time.

    (Not at all, to the rounding of its argument: where the argument jumps
    by one and a half the sawtooth moves by half a wrap, and that is a jump
    of it. Nor more than one double of the clock moves it, four times over:
    at t = 24000 that is 1e-12 of ``frac(0.37*mod(time(), 10) + ...)``,
    through its slope, which is more than any rounding of it.)

    How far one double moves it is the least of what it moves by across a
    few of them, beside the jump and further out. Beside it alone, a jump in
    the noise at the edge of a window had another on the next double, was
    held against that, and was dropped with the edge.
    """
    reading, _, wraps = held
    if wraps is None or at.size == 0:
        return np.ones(at.size, dtype=bool)
    ulp = np.spacing(np.abs(at))
    with np.errstate(invalid="ignore"):
        moved = np.abs(wraps(at) - wraps(np.nextafter(at, -np.inf)))
        drift = np.full(at.size, np.inf)
        for doubles in _DRIFT_AT:
            here = at + doubles * ulp
            step = np.abs(wraps(np.nextafter(here, np.inf)) - wraps(here))
            drift = np.fmin(drift, step)
        drift = np.where(np.isfinite(drift), drift, 0.0)
        # (A quarter of a wrap at the most. Of an argument at 1e15 the doubles
        # are an eighth apart, a touch of them is eight wraps, and every wrap
        # of `frac` of such a line was no jump: no stop on any.)
        rounding = np.minimum(
            TOUCH_ULPS * np.spacing(np.maximum(np.abs(reading(at)), 1.0)), _WRAP_AT_LEAST
        )
        return ~(moved <= np.maximum(rounding, _JUMP_OVER_DRIFT * drift))


def _jumps_on(held: _Held | None, at: np.ndarray) -> np.ndarray:
    """Whether a text jumps on each of the doubles *at*, on which something
    inside it does: its value there is another than on the double before (for
    a sawtooth, by more than it moves from one double to the next). A text
    nothing is known to hold of is taken to jump with what is inside it.

    This is asked on the double, and not of the few doubles around it by
    their two ends. ``frac(floor(time()/0.001) + 0.3*time())`` does not move
    where ``floor`` jumps, and is 1.5e-11 of rounding on every double: across
    a bracket a few doubles long that was more than it drifts, and a stop
    appeared where the argument passes a power of two, on one grid and not
    on the next."""
    if held is None or at.size == 0:
        return np.ones(at.size, dtype=bool)
    if held.wraps is not None:
        return _moves(held, at)
    return _differs(held.reading(at), held.reading(np.nextafter(at, -np.inf)), held.by)


def _jumps_across(
    ev: Evaluate,
    call: str,
    lo: np.ndarray,
    hi: np.ndarray,
    lo_cut: np.ndarray,
    hi_cut: np.ndarray,
) -> np.ndarray:
    """The first double of *call*'s new value inside each bracket ``[lo, hi]``
    it jumps across.

    The call jumps across a bracket where its ends differ by more than it
    moves over a stretch of the bracket's width on either side. A call that
    steps also where its two ends, each held for a few doubles beyond it,
    differ at all, whatever it reads a bracket's width further out: a bracket
    that ends in the middle of a window is as long as from there to the
    window's edge, and the same again on its far side is the window's other
    edge.
    """
    if lo.size == 0:
        return np.empty(0)
    at_lo = ev(call, lo)
    at_hi = ev(call, hi)
    width = hi - lo
    with np.errstate(invalid="ignore"):
        below = np.abs(at_lo - ev(call, lo - width))
        above = np.abs(ev(call, hi + width) - at_hi)
    # A cut end (*lo_cut*, *hi_cut*) has another crossing within the bracket's
    # width of it: what the call does across that is the other crossing's
    # jump and no drift. (`frac(sin(3.644*time() + 2.142) + 6.7e-16)` wraps
    # by one into each top and by one out of it, and each half's wrap, held
    # against the other's, was no jump; so was the wrap of one of two windows
    # 170 time units wide, held against the other window's.)
    below = np.where(lo_cut, 0.0, below)
    above = np.where(hi_cut, 0.0, above)
    with np.errstate(invalid="ignore"):
        drift = np.maximum(below, above)
    # Where the call is not a number just outside, nothing is known of its
    # drift, and the difference of the ends is taken as it stands.
    drift = np.where(np.isfinite(drift), drift, 0.0)
    noise = 4.0 * np.spacing(np.maximum(np.abs(at_lo), np.abs(at_hi)))
    with np.errstate(invalid="ignore"):
        jumps = (
            np.isfinite(at_lo)
            & np.isfinite(at_hi)
            & (np.abs(at_hi - at_lo) > _JUMP_OVER_DRIFT * drift + noise)
        )
        held = _held(ev, call)
        if held is not None and held.wraps is not None and (lo_cut & hi_cut).any():
            # With its drift unread beyond either end, a sawtooth jumps where
            # the whole number it takes off its argument is another at one
            # end than at the other, or where it has moved by a quarter of a
            # wrap, which no drift does across a bracket (its argument has
            # jumped under it: `mod(floor(3*sin(time())), 2)` at a top).
            #
            # (With one end cut, its drift is read beyond the other, and what
            # it moves by is held against that. `frac(0.3*time() + 0.2*(1 +
            # sign(c - q))/2)` for a coarse q goes up by a tenth of a wrap on
            # each side of the tread `sign` is zero for, in two halves of one
            # bracket; asked for a quarter, it had no jump at either edge of
            # the pulse.)
            which = np.nonzero(lo_cut & hi_cut & jumps)[0]
            with np.errstate(invalid="ignore"):
                moved = np.abs(held.wraps(hi[which]) - held.wraps(lo[which])) > 0.25
            jumps[which] = (
                _differs(held.reading(lo[which]), held.reading(hi[which]), held.by) | moved
            )
        if held is not None and held.wraps is None:
            # A call that steps and holds its value for a few doubles beyond
            # each end is on a tread of its stairs at both, and has jumped
            # where the two differ. (Beside a pole it holds nothing: `round`
            # of `1/(time() - a)` is a new whole number on every double there,
            # and its ends are left to how much it drifts.)
            few = _NEIGHBOURS * np.spacing(np.maximum(np.abs(lo), np.abs(hi)))
            holds = (ev(call, lo - few) == at_lo) & (ev(call, hi + few) == at_hi)
            jumps |= np.isfinite(at_lo) & np.isfinite(at_hi) & holds & (at_hi != at_lo)
    if not jumps.any():
        return np.empty(0)
    from_lo, from_hi = at_lo[jumps], at_hi[jumps]

    def nearer_lo(times: np.ndarray, live: np.ndarray) -> np.ndarray:
        values = ev(call, times)
        return np.abs(values - from_lo[live]) <= np.abs(values - from_hi[live])

    return np.unique(
        _bisect(nearer_lo, lo[jumps], hi[jumps], np.ones(int(jumps.sum()), dtype=bool))
    )


def _missed_between(
    f: Sampled,
    held: _Held,
    found: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    lo_clear: np.ndarray,
    hi_clear: np.ndarray,
    rounding: np.ndarray,
    span: tuple[float, float],
    zero_only: bool,
) -> np.ndarray:
    """Jumps of a call that its brackets ``[lo, hi]`` did not find, from what
    the call reads at points where its argument is clear of every level.

    Between two brackets the argument crosses nothing, so what the call holds
    from one jump to the next (:func:`_held`) is the same at any two points
    with no jump found between them. It is read where the argument is known
    to be clear of its rounding: at a bracket's end that got clear before the
    cap, half way from one bracket to the next, and at the ends of the
    window, the last two where the argument is as far from every level as a
    bracket's end is asked to be. Where two such readings in a row differ and
    no jump was found between them, the jump is bisected for where the
    readings say it is. (The caller then asks whether the argument flickers
    there, and refuses the call if it does.)

    A point where the argument is not clear is not read. Half way between two
    crossings of ``3*sin(time())`` through 2 is the top of the sine, which
    rounds to 3 on a stretch of doubles there and makes ``frac`` of it drop to
    zero: read, it gave the top a pulse on one grid, and on the next, where
    the pulse's own edges were crossings with a bracket over them, none.
    """
    reading, by, _ = held

    def far_from_a_level(times: np.ndarray, rounding: np.ndarray) -> np.ndarray:
        value = f(times)
        off = np.abs(value) if zero_only else np.abs(value - np.rint(value))
        with np.errstate(invalid="ignore"):
            return np.isfinite(value) & (off >= _CLEAR_OF_ROUNDING * rounding)

    between = 0.5 * (hi[:-1] + lo[1:])
    points = [lo[lo_clear], hi[hi_clear]]
    if between.size:
        points.append(between[far_from_a_level(between, np.maximum(rounding[:-1], rounding[1:]))])
    for end, beside in ((span[0], 0), (span[1], -1)):
        if (end < lo[0] or end > hi[-1]) and far_from_a_level(np.array([end]), rounding[beside])[
            0
        ]:
            points.append(np.array([end]))
    at = np.unique(np.concatenate(points))
    if at.size < 2:
        return np.empty(0)
    read = reading(at)
    changed = _differs(read[:-1], read[1:], by)
    if found.size:
        # The first jump found after each point, where it is before the next.
        after = np.searchsorted(found, at[:-1], side="right")
        changed &= ~((after < found.size) & (found[np.minimum(after, found.size - 1)] <= at[1:]))
    missed = np.nonzero(changed)[0]
    if missed.size == 0:
        return np.empty(0)
    was = read[missed]

    def as_it_was(times: np.ndarray, live: np.ndarray) -> np.ndarray:
        return ~_differs(reading(times), was[live], by)

    more = _bisect(as_it_was, at[missed], at[missed + 1], np.ones(missed.size, dtype=bool))
    return more[_moves(held, more)]


def _around(times: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """*times* and the doubles on each side of them, inside ``[lo, hi]``."""
    if times.size == 0:
        return times
    out = [times]
    down = up = times
    for _ in range(_NEIGHBOURS):
        down = np.nextafter(down, -np.inf)
        up = np.nextafter(up, np.inf)
        out.extend((down, up))
    near = np.concatenate(out)
    return near[(near >= lo) & (near <= hi)]


def _truth(values: np.ndarray) -> np.ndarray:
    """The truth of a condition's value, with not a number a value of its own."""
    return np.where(np.isnan(values), 2.0, (values != 0.0).astype(float))


def _one_instant(edges: np.ndarray, keep_even: bool) -> np.ndarray:
    """*edges* with each run of changes a few doubles apart read as one
    instant: an odd run is one change (the last of it, from which the new value
    holds), an even run none, unless *keep_even*, which leaves it whole."""
    if edges.size < 2:
        return edges
    close = np.diff(edges) <= TOUCH_ULPS * np.spacing(np.abs(edges[:-1]))
    starts = np.nonzero(np.concatenate(([True], ~close)))[0]
    ends = np.concatenate((starts[1:], [edges.size]))
    size = ends - starts
    keep = np.zeros(edges.size, dtype=bool)
    keep[ends[size % 2 == 1] - 1] = True
    if keep_even:
        for s, e in zip(starts[size % 2 == 0], ends[size % 2 == 0], strict=True):
            keep[s:e] = True
    return edges[keep]


# The golden ratio's fractional part: the points ``frac(i * _GOLDEN)`` fall
# evenly over a period of any length, where those of an even grid all fall on
# a few phases of a period that nearly divides its spacing.
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


@functools.lru_cache(maxsize=16)
def _unit_grid(cells: int) -> tuple[np.ndarray, np.ndarray]:
    """The points of :func:`_grid` on ``[0, 1]``, sorted, for one size, and
    which of them are the even grid's."""
    even = np.linspace(0.0, 1.0, cells + 1)
    unit = np.union1d(even, np.mod(np.arange(1, cells + 1) * _GOLDEN, 1.0))
    at = np.searchsorted(unit, even)
    unit.setflags(write=False)
    at.setflags(write=False)
    return unit, at


def _grid(lo: float, hi: float, cells: int) -> np.ndarray:
    """An even grid of *cells* over ``[lo, hi]`` and as many points again that
    no period divides.

    The even grid lands on the round numbers conditions are written with. On
    its own it reads a fast oscillation through a stroboscope: of
    ``abs(sin(pi*time())) > 0.9999`` over 100,000 time units, grids of 8,192
    and 16,384 cells both found the same 1,696 windows of 100,000, and the
    count having settled was taken for the count being right. The other
    points see a different phase each, so what they find of something not yet
    resolved grows with how many they are, and two passes do not agree.
    """
    unit, at = _unit_grid(cells)
    nodes = lo + (hi - lo) * unit
    # The even points exactly, as `linspace` gives them: a condition written
    # on 120 is read on 120.
    even = np.linspace(lo, hi, cells + 1)
    even[0], even[-1] = lo, hi
    nodes[at] = even
    nodes = nodes[(nodes >= lo) & (nodes <= hi)]
    return nodes[np.concatenate(([True], np.diff(nodes) > 0.0))]


_READS_CLOCK = re.compile(r"(?<![A-Za-z0-9_.])(time|tfun_[A-Za-z0-9_]+)\s*\(")
# One piece of a text made of numbers written out, names, `+`, `-`, `*`,
# parentheses, the clock, and `/` by a number or a name. (Matched piece by
# piece from the left. As one pattern repeated over the whole text, a number of
# seventeen digits can be cut into pieces in 2^16 ways, and each was tried
# before a text with an `exp(` in it was given up: a minute for one call.)
_NUMBER = r"(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
# (A name that is no call, and no part of a dotted one.)
_NAME = r"[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_.]*\s*\()(?!\.)"
_LINE_PIECE = re.compile(
    rf"\s*(?:(?P<clock>time\(\))|/\s*[-+]?\s*(?:{_NUMBER}|(?P<over>{_NAME}))"
    rf"|{_NUMBER}|(?P<name>{_NAME})|[-+()]|\*(?!\s*\*))\s*"
)


def _reads_of_the_clock_in_a_line(
    text: str, constants: Collection[str] = frozenset()
) -> int | None:
    """How many times *text* reads the clock, where it is made of nothing but
    the pieces of a line in time; None where it holds anything else. A name is
    such a piece where it is one of *constants*: what does not move in the
    run."""
    at, clocks = 0, 0
    while at < len(text):
        piece = _LINE_PIECE.match(text, at)
        if piece is None or piece.end() == at:
            return None
        named = piece.group("name") or piece.group("over")
        if named is not None and named not in constants:
            return None
        clocks += piece.group("clock") is not None
        at = piece.end()
    return clocks


def _a_line_in_time(phase: Phase, constants: Collection[str] = frozenset()) -> bool:
    """Whether what *phase* crosses its levels with is the clock times a number
    plus a number, by its spelling: one ``time()``, numbers written out and
    names of *constants* (what the caller knows not to move in the run),
    ``+``, ``-``, ``*``, parentheses, and ``/`` by a number or such a name (and
    for a call that compares two of its arguments, one of them such a text and
    the other with no clock in it).

    Each of those operations gives the double nearest its result, and the
    nearest double to a sum, a product or a quotient with a constant does not
    come down as the other operand goes up, nor up as it comes down. So
    neither does the whole text: it never steps back, and crosses a level
    once, however many of its own last places it stays beside it for.
    ``1e12 + time()`` moves once in 500 instants of a run 240 long, and is
    never clear of a whole number it has just crossed for sixteen of those
    moves; what is true of a sum of two such terms, which does step back
    (``(1.77*time() + 1e9) - 0.68*time()``), is not true of it.
    """
    # The calls read through the size of their argument (`abs(x)`, which is
    # not a line): `trunc`, `frac` and `round` of a line, and `mod` of a line
    # by numbers. The size of a line comes down to its zero and goes up from
    # it, and a bracket is read by its ends only with one crossing in it
    # (`_brackets`): one that held the turn and a change on each side of it
    # would hold two.
    #
    # (`mod(x, m)` is the remainder exactly, and its phase is the quotient
    # rounded: its wrap is not always on one of the two doubles the phase
    # crosses a level on. What the call holds is read between the brackets
    # as well for a line, which is where that wrap then is:
    # :func:`_call_jumps`.)
    name, _, rest = phase.call.partition("(")
    if name.strip() in ("trunc", "frac", "round", "mod"):
        reads = [
            _reads_of_the_clock_in_a_line(arg.strip(), constants)
            for arg in _split_commas(rest[:-1])
        ]
        return reads == ([1, 0] if name.strip() == "mod" else [1])
    if phase.sides is not None:
        sides = [_reads_of_the_clock_in_a_line(side, constants) for side in phase.sides]
        return sorted(-1 if n is None else n for n in sides) == [0, 1]
    return _reads_of_the_clock_in_a_line(phase.phase, constants) == 1


def _residual(ev: Evaluate, lhs: str, rhs: str, lo: float) -> Sampled:
    """``lhs - rhs`` over times, each side evaluated on its own. A side that
    reads no clock (the threshold, as a rule) is evaluated once."""

    def side(text: str) -> Sampled:
        if _READS_CLOCK.search(text):
            return functools.partial(ev, f"({text})")
        value = float(ev(f"({text})", np.array([lo]))[0])
        return lambda at: np.full(at.shape, value)

    left, right = side(lhs), side(rhs)
    return lambda at: left(at) - right(at)


# The whole number a sawtooth (`mod`, `frac`) may take off its argument: 2^48,
# where the doubles of the quotient are a sixteenth of a level apart. Beyond
# it a period is a dozen doubles of the argument or fewer, and what the call
# returns is the argument's rounding: it goes up in a few steps, each a
# sizable part of the modulus, and comes down by the rest. A stop on each wrap
# is not a stop on each step. With a modulus of 12 doubles of `1e12 + time()`
# every wrap had its stop and X came to 0.72 of the law's integral, where main,
# with no stop at all, has it right; with one of 1.2 doubles no step of the
# call stands out as a wrap, and there was no stop and no word.
_SAWTOOTH_AT_MOST = 2.0**48


def _start(lo: float, hi: float) -> float:
    """The time up to which a change is the run's own start, and no stop.

    A few doubles of the window's scale, not of its first instant. At
    ``lo = 0`` a few doubles are 3e-322, and ``time()/1000`` underflows to
    zero on the first of them: `max(0, time()/1000)` turned at 2.5e-321,
    which CVODE cannot step to, and the run returned zeros or died.
    """
    return lo + TOUCH_ULPS * float(np.spacing(max(abs(lo), abs(hi))))


def _scan(
    ev: Evaluate,
    text: str,
    lo: float,
    hi: float,
    cells: int,
    kind: str,
    sides: tuple[str, str] | None,
    knots: Sequence[float],
    budget: int,
    touches: bool,
    instant: float,
    constants: Collection[str] = frozenset(),
) -> np.ndarray:
    """One pass of :func:`scan_changes`, on a grid of *cells*."""
    nodes = _grid(lo, hi, cells)
    if len(knots):
        k = np.asarray(knots, dtype=float)
        nodes = _merge(nodes, _around(k[(k >= lo) & (k <= hi)], lo, hi))

    jumps = np.empty(0)
    turns = np.empty(0)
    counted = 0
    zones_of: dict[str, list] = {}
    table_knots = np.unique(np.asarray(knots, dtype=float)) if len(knots) else None
    for phase in phases(text):
        f = (
            _residual(ev, *phase.sides, lo)
            if phase.sides is not None
            else functools.partial(ev, phase.phase)
        )
        if phase.sides is not None and phase.call.partition("(")[0].strip() == "round":
            f = _phase_of_round(ev, phase.sides[0])
        # (Of a sawtooth that is the text itself. Under a comparison or inside
        # a call that steps, each change of the text has its stop, however
        # few doubles a wrap is: `if(mod(1e15 + time(), 2) < 1, 1, 0)` is 120
        # for 240 with its 240 stops, and was refused with the rest. And of a
        # quotient that is a number: a law guarded against the one point its
        # argument is not (`if(time() > 0.5, frac(ln(time())), 0)` from 0) is
        # read.)
        alone = kind == "value" and phase.call == text
        if alone and phase.call.partition("(")[0].strip() in _SAWTOOTH:
            size = np.abs(f(np.append(nodes[:: max(1, nodes.size // 64)], nodes[-1])))
            if np.any(size[np.isfinite(size)] >= _SAWTOOTH_AT_MOST):
                raise Flickers(phase.call, lo, hi, unread=True, coarse=True)
        try:
            crossed, domain = _crossings(f, nodes, phase.zero_only, budget - counted)
        except OverBudget as exc:
            # With what the calls before this one came to, which is the count
            # the caller reports.
            raise OverBudget(counted + exc.count) from None
        counted += crossed.size
        nodes = _merge(nodes, _around(domain, lo, hi))
        if phase.bends:
            turns = np.union1d(turns, crossed)
            nodes = _merge(nodes, _around(crossed, lo, hi))
            # ...and a point between each two turns. A pulse written with a
            # bend is flat outside its turns, and a step call of it changes
            # between them: `round(max(0, 1 - ((time() - 100.01)/0.001)^2))`
            # is one for 0.0014 with a grid point at neither foot of the
            # bump nor inside it, and had stops at the feet alone.
            nodes = _merge(nodes, 0.5 * (turns[:-1] + turns[1:]))
        else:
            # (Where the calls inside this one's argument were read for their
            # jumps, each before it: `phases` gives the innermost first.)
            read_in = phase.phase if phase.sides is None else ",".join(phase.sides)
            inner = [inside for inside in zones_of if inside in read_in]
            held_inside = [held for held in (_held(ev, inside) for inside in inner) if held]
            held_inside += [
                _Held(functools.partial(ev, f"({compared})"), 0.0, None)
                for compared in _comparisons(read_in)
            ]
            line = _a_line_in_time(phase, constants)
            around = (
                _brackets(
                    f,
                    crossed,
                    phase.zero_only,
                    instant,
                    _zones([z for inside in inner for z in zones_of[inside]]),
                    held_inside,
                    table_knots,
                    line,
                )
                if crossed.size
                else None
            )
            mine: list = []
            jumped = _call_jumps(
                ev,
                phase.call,
                f,
                crossed,
                (lo, hi) if kind == "value" else None,
                phase.zero_only,
                instant,
                brackets=around,
                within=(lo, hi),
                zones=mine,
                line=line,
            )
            zones_of.setdefault(phase.call, []).extend(mine)
            nodes = _merge(nodes, _around(jumped, lo, hi))
            if kind == "value":
                # What the text itself does where this call jumps: the call's
                # own jump where the text is that call, and for a call inside
                # it (`floor(floor(time())/3)`), each of its jumps the text
                # changes on.
                own = jumped if phase.call == text else jumped[_jumps_on(_held(ev, text), jumped)]
                jumps = np.union1d(jumps, own)

    start = _start(lo, hi)
    if kind == "bend":
        turns = turns[(turns > start) & (turns <= hi)]
        if turns.size > 1:
            # `abs(u)` with u zero on a double turns there, and the sign of u
            # changes again where it leaves zero: one turn, not two.
            again = np.diff(turns) <= TOUCH_ULPS * np.spacing(np.abs(turns[:-1]))
            turns = turns[np.concatenate(([True], ~again))]
        return turns
    if kind == "value":
        # A step call standing as a number changes where its value jumps, and
        # the bracket around each crossing of its argument has said whether it
        # does: its two ends differ. The rounding of an argument makes the
        # call flicker between the two values for a few doubles (-8, -9, -8,
        # -9, ...), and a jump asked to stand out from its neighbours there
        # did not, and was dropped with the pulse it bounded.
        edges = np.unique(jumps[(jumps > start) & (jumps <= hi)])
        if edges.size > 1:
            # Two calls of the text can each find the one jump they share, a
            # few doubles apart where the text flickers around it: one stop,
            # not two, or the count differs from one grid to the next by
            # which of them the bisection happened to separate.
            again = np.diff(edges) <= TOUCH_ULPS * np.spacing(np.abs(edges[:-1]))
            edges = edges[np.concatenate(([True], ~again))]
        if edges.size > budget:
            raise OverBudget(edges.size)
        return edges

    # A point inside every stretch between two points so far: a piece between
    # two jumps a fraction of a cell apart has no grid point in it, and a
    # window that opens and closes inside it is the same at both of its ends.
    between = nodes[:-1] + 0.5 * (nodes[1:] - nodes[:-1])
    both = np.empty(2 * nodes.size - 1)
    both[0::2] = nodes
    both[1::2] = between
    # (Between two adjacent doubles the point is one of them.)
    nodes = both[np.concatenate(([True], np.diff(both) > 0.0))]

    if kind == "comparison" and sides is not None:
        # Each side evaluated on its own and the difference taken here. Handed
        # `(time()-1)-(1.3)` whole, the evaluator folds the constants into
        # `time()-2.3`, which is not the arithmetic the comparison's sides have.
        lhs, rhs = sides
        residual = _residual(ev, lhs, rhs, lo)
        try:
            crossed, domain = _crossings(residual, nodes, True, budget - counted)
        except OverBudget as exc:
            raise OverBudget(counted + exc.count) from None
        nodes = _merge(nodes, _around(np.union1d(crossed, domain), lo, hi))

    truth = _truth(ev(text, nodes))
    changed = np.nonzero(truth[:-1] != truth[1:])[0]
    if changed.size > budget:
        raise OverBudget(changed.size)

    def side(times: np.ndarray, live: np.ndarray) -> np.ndarray:
        return _truth(ev(text, times))

    edges = _bisect(side, nodes[changed], nodes[changed + 1], truth[changed])
    edges = np.unique(edges[(edges > start) & (edges <= hi)])
    if edges.size > budget:
        raise OverBudget(edges.size)
    return _one_instant(edges, keep_even=touches)


def _instant(lo: float, hi: float) -> float:
    """How far apart two times in the window ``[lo, hi]`` are and still one
    instant: a billionth of the window, or a few thousand doubles of its
    scale. No integrator step is that short, and nothing a run returns can
    tell two changes that close from one."""
    return max(1e-9 * (hi - lo), 4096.0 * float(np.spacing(max(abs(lo), abs(hi)))))


def _instants(edges: np.ndarray, lo: float, hi: float) -> int:
    """How many separate instants *edges* are, two closer than a billionth of
    the window (or a few thousand doubles) being one.

    What two passes are asked to agree on. Not the count of edges: around a
    crossing the text flickers for tens of doubles, and how many changes a
    pass reports from inside that stretch (one, or three, or one from each of
    two calls that share the jump) depends on where its points fell. Counted
    edge by edge, a condition with 5,507 changes was found with 5,507 and
    5,508 by turns and refused as one whose changes could not be counted.
    """
    if edges.size == 0:
        return 0
    return 1 + int(np.count_nonzero(np.diff(edges) > _instant(lo, hi)))


def scan_changes(
    ev: Evaluate,
    text: str,
    lo: float,
    hi: float,
    *,
    kind: str,
    sides: tuple[str, str] | None = None,
    knots: Sequence[float] = (),
    budget: int,
    touches: bool = False,
    instant: float | None = None,
    constants: Collection[str] = frozenset(),
) -> list[float]:
    """Each time in ``(lo, hi]`` at which *text* changes, as the first double
    at which ``ev`` reads the new value.

    *kind* says what a change is:

    * ``"comparison"``: *text* is ``lhs op rhs`` and *sides* its two sides. A
      change is a change of its truth.
    * ``"truth"``: *text* is a condition with no comparison of its own
      (anything non-zero is true). A change is a change of its truth, which is
      all an ``if()`` reads of it.
    * ``"value"``: *text* is a step call standing in a rate law outside any
      condition. A change is a jump of its value: ``mod(time(), P)`` is zero
      at one double a period, which is no window.
    * ``"bend"``: *text* is a call that bends (``abs``, ``min``, ``max``,
      ``clamp``) standing in a rate law the same way. A change is a turn of
      one of the bending calls in it.

    *constants* are the names in the text that do not move in the window (a
    parameter no event or rule writes): a step call of the clock times such a
    name plus another is read as one of the clock times a number is
    (:func:`_a_line_in_time`).

    *knots* are times the text is known to bend or jump at (a table function's
    breakpoints), taken as grid points. A change within a few doubles of *lo*,
    on the scale of the window, is the run's own start, and is not returned.
    Changes a few doubles apart are one instant: an odd number of them one
    change, an even number none (or, with *touches*, for an event trigger, all
    of them).

    Raises :class:`OverBudget` past *budget* changes, :class:`Unresolved`
    where two grids, one twice as fine as the other, still disagree about how
    many there are at :data:`MAX_GRID_CELLS`, and :class:`Flickers` where a
    step call's argument is within its rounding of a level it crosses for
    longer than *instant*: a billionth of the window, unless the window is a
    piece of a run and the caller passes the run's.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return []
    if instant is None:
        instant = _instant(lo, hi)
    cells = GRID_CELLS
    coarse = _scan(
        ev, text, lo, hi, cells, kind, sides, knots, budget, touches, instant, constants
    )
    while True:
        cells *= 2
        # The changes of the pass before are points of this one, so that it
        # looks for what that pass missed and does not search again for what
        # it found. (Not where they are too many to be worth sorting in.)
        #
        # And the middle between each two of them. The two edges of a pulse
        # can be all that is left of it as grid points: `round(sin(3.002*
        # time() + 0.464) - 2.7e-16 - 0.5)` is -2 for 1.9e-8 at each bottom,
        # its argument reads the same on the first double of the pulse and on
        # the last, with nothing between them to say it dips, and the pass
        # after the one that found the pulses found none: 19 changes, 9, 19,
        # 9, and the condition was refused as one that could not be counted.
        known = (
            (*knots, *coarse, *(0.5 * (coarse[:-1] + coarse[1:])))
            if coarse.size <= cells
            else knots
        )
        fine = _scan(
            ev, text, lo, hi, cells, kind, sides, known, budget, touches, instant, constants
        )
        if _instants(fine, lo, hi) == _instants(coarse, lo, hi):
            return [float(x) for x in fine]
        if cells >= MAX_GRID_CELLS:
            raise Unresolved(max(fine.size, coarse.size))
        coarse = fine


# Sample points of :func:`few_changes`, and how many cells around a change of
# the comparison it will scan in full.
_FEW_SAMPLES = 1 << 14
_FEW_ZONE_CELLS = 64


def few_changes(
    ev: Evaluate, text: str, lo: float, hi: float, sides: tuple[str, str], *, budget: int
) -> list[float] | None:
    """The changes of a comparison whose step calls jump more often than a run
    has stops for, where the comparison can be seen not to follow them, or
    ``None``.

    ``floor(time()/0.0001) < 5000`` holds 2.4 million jumps in 240 time units
    and changes once; ``mod(time(), 1e-4) < 1`` holds as many and never
    changes. Counting the jumps refused both.

    The difference of the two sides is sampled, on an even grid and on a
    sequence that no period divides. Where it stays further from zero than
    twice the most it moves between two samples, the comparison is taken not
    to change. The few cells where it comes nearer are scanned in full
    (:func:`scan_changes`), which is exact there or over the budget in its own
    right. A comparison whose difference is near zero over many cells (a
    sawtooth against a threshold inside its range) is left to the caller.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return []
    # A step call inside another flattens what it does: `sign(floor(u) + 0.5)`
    # is one wherever u is not negative, however u moves, and minus one for
    # as long as u dips under zero between two samples, with nothing in the
    # samples to say so. Nor is anything said by a difference that does not
    # move at all from one sample to the next.
    #
    # Nor by another step call beside it. `floor(time()/0.0001)*1e-7 +
    # 5*(sign(time() - 99.9917) - sign(time() - 100.0034)) > 3` is true for
    # 0.0117 between two samples 0.0146 apart, and the samples read none of
    # the two `sign` calls that make it so. So: one step call, and no other.
    # (A call that bends beside it jumps nowhere: `abs(floor(time()*1e4)/1e4
    # - 120) < 0.25` follows its staircase.)
    if len({phase.call for phase in phases(text) if not phase.bends}) != 1:
        return None
    nodes = _grid(lo, hi, _FEW_SAMPLES)
    lhs, rhs = sides
    residual = _residual(ev, lhs, rhs, lo)(nodes)
    if not np.all(np.isfinite(residual)):
        return None
    truth = _truth(ev(text, nodes))
    reach = 2.0 * float(np.abs(np.diff(residual)).max())
    if reach == 0.0:
        return None
    near = np.abs(residual) <= reach
    cell = near[:-1] | near[1:] | (truth[:-1] != truth[1:])
    zone = np.nonzero(cell)[0]
    if zone.size == 0:
        return []
    if zone.size > _FEW_ZONE_CELLS:
        return None
    edges: list[float] = []
    runs = np.split(zone, np.nonzero(np.diff(zone) > 1)[0] + 1)
    for run in runs:
        # A cell further on each side, in which nothing changes.
        a = nodes[max(int(run[0]) - 1, 0)]
        b = nodes[min(int(run[-1]) + 2, nodes.size - 1)]
        try:
            # (What flickers is judged by an instant of the run, not of this
            # piece of it: a floor of an argument coarser than the clock reads
            # both values for a few thousand doubles at each step, which is a
            # billionth of a piece a few cells long and nothing of the run.)
            edges.extend(
                scan_changes(
                    ev,
                    text,
                    a,
                    b,
                    kind="comparison",
                    sides=sides,
                    budget=budget,
                    instant=_instant(lo, hi),
                )
            )
        except OverBudget:
            return None
    start = _start(lo, hi)
    return sorted({x for x in edges if start < x <= hi})
