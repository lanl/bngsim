"""Issue #1518: ``steady_states()`` tells roots apart by a distance in
concentration, not by its residual tolerance.

``SteadyStates.cluster`` merged two roots when every species agreed to
``dedupe_rtol`` of its magnitude plus ``tol`` as an absolute floor, and
``tol`` (default 1e-9) is a residual in concentration per time. In molar units
at nanomolar scale the roots of a switch 0.3 nM apart were one root, and the
seed decided whether the attractor or the saddle survived. The floor is now
``SteadyStates.ROUNDOFF`` (1e-9) of the largest value each species takes at any
root, the round-off a solve leaves in a species at zero, and not under double
precision's resolution beside the largest concentration.

Oracles: the Schlogl switch below is built so that dX/dt is a cubic with roots
at exactly 0.2 s, 0.5 s and 3 s, stable, unstable, stable by the sign of its
slope, at every scale s; the clustering cases are hand-written states whose
distances are chosen on either side of the floor. Newton places a simple root
to far better than 1e-6 of its value at rtol 1e-10 / atol 1e-12 (the issue's
table agrees to 4 digits at both scales), so X/s is compared at rel 1e-6;
stability labels and root counts are exact.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

# A + 2X <-> 3X, X <-> B, A and B $-fixed at s. With k1..k4 as below,
# dX/dt = -(1/(tau s^2)) (X - 0.2s)(X - 0.5s)(X - 3s): roots at 0.2s (stable),
# 0.5s (unstable) and 3s (stable) at every s.
ROOTS = [(0.2, "stable"), (0.5, "unstable"), (3.0, "stable")]


def schlogl_net(s: float, tau: float = 0.1) -> str:
    a, b, d = 0.2 * s, 0.5 * s, 3.0 * s
    c = 1.0 / (tau * s * s)
    return f"""begin parameters
    1 k1 {c * (a + b + d) / s!r}
    2 k2 {c!r}
    3 k3 {c * (a * b + a * d + b * d)!r}
    4 k4 {c * a * b * d / s!r}
end parameters
begin species
    1 $A() {s!r}
    2 $B() {s!r}
    3 X() {1.5 * s!r}
end species
begin reactions
    1 1,3,3 3,3,3 k1
    2 3,3,3 1,3,3 k2
    3 3 2 k3
    4 2 3 k4
end reactions
"""


def roots_of(sim: bngsim.Simulator, s: float, **kw) -> list[tuple[float, str]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        roots = sim.steady_states(**kw)
    order = np.argsort(roots.states[:, 2])
    return [(float(roots.states[i, 2] / s), roots.stability[i]) for i in order]


@pytest.mark.parametrize("seed", [0, 2])
@pytest.mark.parametrize("s", [1.0, 1e-9, 1e3], ids=["unit-scale", "nanomolar", "1e3"])
def test_a_nanomolar_switch_keeps_its_three_roots(tmp_path, s, seed):
    """At s = 1e-9 the default call reported two roots: seed 0 dropped the
    0.2 nM attractor and seed 2 the 0.5 nM saddle, each merged into the other
    because |dX| = 0.3 nM <= 1e-6 |X| + tol with tol = 1e-9."""
    path = tmp_path / "schlogl.net"
    path.write_text(schlogl_net(s))
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    got = roots_of(sim, s, seed=seed)
    assert [stab for _, stab in got] == [stab for _, stab in ROOTS]
    assert [x for x, _ in got] == pytest.approx([x for x, _ in ROOTS], rel=1e-6)


def test_the_merge_distance_does_not_follow_tol(tmp_path):
    """A tighter tol gave all three roots before (the issue's last row); now tol
    changes nothing about how roots are told apart."""
    s = 1e-9
    path = tmp_path / "schlogl.net"
    path.write_text(schlogl_net(s))
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    default = roots_of(sim, s, seed=0)
    tight = roots_of(sim, s, seed=0, tol=1e-12)
    assert [stab for _, stab in default] == [stab for _, stab in tight]
    assert [x for x, _ in default] == pytest.approx([x for x, _ in tight], rel=1e-6)


# ── SteadyStates.cluster on hand-written states ──────────────────────────────


class _Converged:
    def __init__(self, y, residual=1e-12):
        self.converged = True
        self.concentrations = np.asarray(y, float)
        self.residual = residual
        self.root_stability = "stable"
        self.eigenvalues = np.array([-1.0])


def _cluster(states, **kw):
    return bngsim.SteadyStates.cluster(
        [_Converged(y) for y in states], ["P", "X"], dedupe_rtol=1e-6, **kw
    )


def test_roots_a_residual_apart_are_distinct():
    """The issue's grouping: X at 0.2 nM and 0.5 nM beside a pool at 1 nM. With
    the floor derived from the roots (1e-9 of 1e-9 M), two roots; with the old
    floor, the residual 1e-9, one."""
    states = [[1e-9, 0.2e-9], [1e-9, 0.5e-9]]
    assert len(_cluster(states)) == 2
    assert len(_cluster(states, atol=1e-9)) == 1


@pytest.mark.parametrize("s", [1.0, 1e-9, 1e-14], ids=["unit", "nano", "1e-14"])
def test_a_root_at_zero_is_one_root_at_any_scale(s):
    """A root at 0 is reached as a few 1e-15 of the species' scale on either side
    of it (the cubic of #1001 landed at -5.9e-24 for s = 1e-9, beside its roots
    at 0.3 s and s). Those land in one root, since the floor is 1e-9 of the
    species' own scale, not 0 (which splits them: the relative term has no
    magnitude to work with)."""
    states = [[s, s], [s, 0.0], [s, -5.9e-15 * s], [s, 3.0e-15 * s]]
    assert len(_cluster(states)) == 2
    assert len(_cluster(states, atol=0.0)) == 4


@pytest.mark.parametrize("s", [1.0, 1e-9, 1e-14], ids=["unit", "nano", "1e-14"])
def test_the_floor_is_roundoff_of_the_species_own_scale(s):
    """X takes s at one root. Two more roots whose X differ by half of 1e-9 s
    merge; by twice it, they stay apart; the floor follows X's scale, and the
    pool P, a thousand times larger, does not set it."""
    floor = bngsim.SteadyStates.ROUNDOFF * s
    assert floor == 1e-9 * s
    big = 1e3 * s
    assert len(_cluster([[big, s], [big, 0.0], [big, 0.5 * floor]])) == 2
    assert len(_cluster([[big, s], [big, 0.0], [big, 2.0 * floor]])) == 3


def test_a_small_switch_beside_a_large_pool_keeps_its_roots():
    """#1265's shape: roots 1e-4 apart in a pool of 1e-3 beside a pool of 1e8.
    A floor taken from the model's largest concentration (0.1) merged them;
    the species' own scale (6.4e-13) keeps them apart."""
    states = [[1e8, 3.862e-5], [1e8, 2.6388e-4], [1e8, 6.3959e-4]]
    assert len(_cluster(states)) == 3


def test_a_species_with_no_scale_of_its_own_takes_the_precisions_beside_the_largest():
    """A species at zero in every root is told apart only beyond double
    precision's resolution beside the largest magnitude: beside a pool of 1e8
    that is ε·1e8 = 2.2e-8. 1e-9 of that, the floor as first written, is below
    the round-off Newton leaves at 0 (see the origin-only model below)."""
    eps = np.finfo(float).eps
    assert len(_cluster([[1e8, 0.0], [1e8, 0.5 * eps * 1e8]])) == 1
    assert len(_cluster([[1e8, 0.0], [1e8, 2.0 * eps * 1e8]])) == 2


def test_scale_sets_the_floor_where_every_value_is_round_off():
    """Where every species is at 0 at every root, the results hold only
    round-off and give no scale: read from them alone, values 1e-30 either side
    of 0 are two roots. ``scale``, the largest magnitude of the starts, gives
    the floor ε·scale, and they are one."""
    states = [[0.0, 1e-30], [0.0, -1e-30]]
    assert len(_cluster(states)) == 2
    assert len(_cluster(states, scale=1.0)) == 1


def test_a_given_atol_is_used_as_given():
    """The public ``atol`` stays a concentration floor the caller sets: 1e-12 M
    keeps 0.2 nM and 0.5 nM apart and still merges the round-off at zero."""
    assert len(_cluster([[1e-9, 0.2e-9], [1e-9, 0.5e-9]], atol=1e-12)) == 2
    assert len(_cluster([[1e-9, 0.0], [1e-9, -5.9e-24]], atol=1e-12)) == 1


def test_no_converged_result_gives_no_roots():
    """An empty or wholly failed batch derives no scale and raises nothing."""
    failed = _Converged([1.0, 1.0])
    failed.converged = False
    assert len(bngsim.SteadyStates.cluster([failed], ["P", "X"], dedupe_rtol=1e-6)) == 0
    assert len(bngsim.SteadyStates.cluster([], ["P", "X"], dedupe_rtol=1e-6)) == 0


# ── The switch on a conservation law, a root at 0, and a pool far larger ─────


def law_switch_net(s: float, tau: float = 0.1) -> str:
    """The same cubic on a conservation law: P + 2Pa -> 3Pa, P <-> Pa, with
    P + Pa = T. With P = T - Pa, dPa/dt = -k1 Pa^3 + k1 T Pa^2 - (k0 + k2) Pa
    + k0 T, which is -(1/(tau s^2)) (Pa - 0.2s)(Pa - 0.5s)(Pa - 3s) when
    T = a + b + d, k0 + k2 = k1 (ab + ad + bd) and k0 T = k1 abd. Stability is
    that of the system reduced by the law, the cubic's slope."""
    a, b, d = 0.2 * s, 0.5 * s, 3.0 * s
    total = a + b + d
    k1 = 1.0 / (tau * s * s)
    k0 = k1 * a * b * d / total
    k2 = k1 * (a * b + a * d + b * d) - k0
    return f"""begin parameters
    1 k1 {k1!r}
    2 k0 {k0!r}
    3 k2 {k2!r}
end parameters
begin species
    1 P() {total - 1.5 * s!r}
    2 Pa() {1.5 * s!r}
end species
begin reactions
    1 1,2,2 2,2,2 k1
    2 1 2 k0
    3 2 1 k2
end reactions
"""


def _law_switch(tmp_path, s: float) -> bngsim.Simulator:
    path = tmp_path / "law_switch.net"
    path.write_text(law_switch_net(s))
    return bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")


def _steady_states(sim: bngsim.Simulator, **kw) -> bngsim.SteadyStates:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sim.steady_states(**kw)


@pytest.mark.parametrize("seed", [0, 2])
@pytest.mark.parametrize("s", [1.0, 1e-9, 1e3], ids=["unit-scale", "nanomolar", "1e3"])
def test_a_switch_on_a_conservation_law_keeps_its_three_roots(tmp_path, s, seed):
    """Pa is a law member, so the law's own floor applied to it, and tol was
    added on top: at s = 1e-9 both seeds gave the two attractors and lost the
    saddle. The ordinary solve's root joins the draws' roots and is no draw."""
    roots = _steady_states(_law_switch(tmp_path, s), seed=seed)
    order = np.argsort(roots.states[:, 1])
    assert [roots.stability[k] for k in order] == [stab for _, stab in ROOTS]
    assert roots.states[order, 1] / s == pytest.approx([x for x, _ in ROOTS], rel=1e-6)
    assert int(roots.multiplicity.sum()) + roots.n_failed == 200


def test_a_start_that_stops_short_is_solved_again(tmp_path, monkeypatch):
    """The switch on a law at s = 1e-9, seed 0: one start runs KINSOL's
    iterations out 0.5% short of the 0.2 s root, at a residual under tol, and
    is accepted there, at a Newton distance of about 4e4 where 1 is within its
    tolerance. tol as the floor had merged it into the root; at the species'
    own scale it was a fourth root. It is solved again from where it stopped,
    and joins the root it fell short of."""
    sim = _law_switch(tmp_path, 1e-9)
    batches = []
    batch = bngsim.Simulator.steady_state_batch

    def spy(self, *args, **kw):
        out = batch(self, *args, **kw)
        batches.append(out)
        return out

    monkeypatch.setattr(bngsim.Simulator, "steady_state_batch", spy)
    roots = _steady_states(sim, seed=0)
    (draws,) = [b for b in batches if len(b) == 200]
    # The witness: a start stopped short. Without one this tests nothing.
    assert any(r.converged and r.newton_distance > 1.0 for r in draws)
    assert len(roots) == 3, roots
    assert np.sort(roots.states[:, 1]) / 1e-9 == pytest.approx([x for x, _ in ROOTS], rel=1e-6)


def root_at_zero_net(s: float, tau: float = 0.01) -> str:
    """dX/dt = -(1/(tau s^2)) X (X - 0.3 s)(X - s): roots 0 and s stable, 0.3 s
    not. tau = 0.01 keeps the starts' residuals above tol = 1e-9 at s = 1e-9,
    past the small-scale refusal."""
    k = 1.0 / (tau * s * s)
    return f"""begin parameters
    1 k3 {k!r}
    2 k2 {k * 1.3 * s!r}
    3 k1 {k * 0.3 * s * s!r}
end parameters
begin species
    1 X() {0.5 * s!r}
end species
begin reactions
    1 1,1,1 1,1 k3
    2 1,1 1,1,1 k2
    3 1 0 k1
end reactions
"""


@pytest.mark.parametrize("s", [1.0, 1e-9], ids=["unit-scale", "nanomolar"])
def test_a_model_with_a_root_at_zero_keeps_it_as_one_root(tmp_path, s):
    """Through steady_states(): at s = 1e-9 the three roots lie within 1e-9 of
    each other, and tol as the floor merged all of them into one. Newton
    reaches the root at 0 at 0 or a hair below it (to -6.7e-34 here), and with
    no floor that root came back as about fifty. The root at 0 is checked to
    1e-12 of s, far inside X's floor of 1e-9 s."""
    path = tmp_path / "root_at_zero.net"
    path.write_text(root_at_zero_net(s))
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    roots = _steady_states(sim, seed=0)
    order = np.argsort(roots.states[:, 0])
    x = roots.states[order, 0] / s
    assert [roots.stability[k] for k in order] == ["stable", "unstable", "stable"]
    assert abs(x[0]) <= 1e-12
    assert x[1:] == pytest.approx([0.3, 1.0], rel=1e-6)


def test_a_switch_beside_a_far_larger_species_keeps_its_roots(tmp_path):
    """The model-level form of #1265's shape: the switch at s = 1e-3 beside a
    species Z held at 1e8 by 0 -> Z and Z -> 0, which shares no reaction or law
    with it. The floor is X's own: 1e-9 of the model's largest value (0.1
    here) would merge all three roots into one, and ε of the largest start,
    2.2e-7, is far below their 3e-4 spacing. decades=13 and 400 starts reach
    X's roots from a sampling bound of 1e9."""
    s = 1e-3
    text = schlogl_net(s)
    text = text.replace("end parameters", "    5 kz 1e8\n    6 dz 1.0\nend parameters")
    text = text.replace("end species", "    4 Z() 1e8\nend species")
    text = text.replace("end reactions", "    5 0 4 kz\n    6 4 0 dz\nend reactions")
    path = tmp_path / "schlogl_beside_z.net"
    path.write_text(text)
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    assert roots_of(sim, s, seed=0, decades=13, n_starts=400) == [
        (pytest.approx(x, rel=1e-6), stab) for x, stab in ROOTS
    ]


def test_a_farther_second_answer_is_not_kept(tmp_path, monkeypatch):
    """The re-solve keeps its second answer only where it is nearer a root.
    Here the second solve of the stalled start is made to answer a state that
    is no root (Pa = s) at a larger Newton distance; the first answer stands,
    and no root appears at that state."""
    s = 1e-9
    sim = _law_switch(tmp_path, s)
    total = (0.2 + 0.5 + 3.0) * s
    draws: list = []
    batch = bngsim.Simulator.steady_state_batch

    class _Farther:
        converged = True
        residual = 0.0
        newton_distance = 1e12
        root_stability = None
        eigenvalues = np.array([])
        concentrations = np.array([total - s, s])

    def spy(self, *args, **kw):
        out = batch(self, *args, **kw)
        if len(out) == 200:
            draws.extend(out)
            return out
        short = [r.concentrations for r in draws if r.converged and r.newton_distance > 1.0]
        states = kw.get("states")
        resolving = (
            short
            and states is not None
            and len(states) == len(short)
            and all(np.array_equal(a, b) for a, b in zip(states, short, strict=True))
        )
        return [_Farther() for _ in out] if resolving else out

    monkeypatch.setattr(bngsim.Simulator, "steady_state_batch", spy)
    roots = _steady_states(sim, seed=0)
    assert any(r.converged and r.newton_distance > 1.0 for r in draws)  # the witness
    pa = roots.states[:, 1] / s
    assert not np.any(np.isclose(pa, 1.0, rtol=1e-6, atol=0.0)), pa
    # The first answer stands: the start it fell short with is still its own root.
    assert len(roots) == 4, pa


def test_a_second_solve_that_raises_leaves_the_first_answers(tmp_path, monkeypatch):
    """Where the re-solve raises, the first answers stand and the call returns,
    as where the seed's solve raises: the stalled start is its own root."""
    sim = _law_switch(tmp_path, 1e-9)
    calls: list[int] = []
    batch = bngsim.Simulator.steady_state_batch

    def spy(self, *args, **kw):
        out = batch(self, *args, **kw)
        calls.append(len(out))
        if len(calls) == 3:  # draws, the seed, then the re-solve
            raise bngsim.SimulationError("refused")
        return out

    monkeypatch.setattr(bngsim.Simulator, "steady_state_batch", spy)
    roots = _steady_states(sim, seed=0)
    assert calls[:2] == [200, 1]
    assert len(roots) == 4, roots


def origin_only_net(s: float) -> str:
    """X -> 0, 2X <-> Y, Y -> 0: every species decays, nothing is made, and the
    one root is the origin. The dimerization makes Newton nonlinear, so it
    reaches 0 with round-off, not exactly (to 7.8e-33 at s = 1e-9)."""
    return f"""begin parameters
    1 a 1.0
    2 b {2.0 / s!r}
    3 c 0.5
end parameters
begin species
    1 X() {1.0 * s!r}
    2 Y() {2.0 * s!r}
end species
begin reactions
    1 1 0 a
    2 1,1 2 b
    3 2 1,1 c
    4 2 0 a
end reactions
"""


@pytest.mark.parametrize("s", [1.0, 1e-9], ids=["unit-scale", "nanomolar"])
def test_a_model_whose_only_root_is_the_origin_has_one_root(tmp_path, s):
    """Main gave one root. A floor read from the results alone, where every
    value is round-off at 0, gave 2 at s = 1 and 108 at s = 1e-9; the
    starts' scale gives one root, at the origin to ε of the largest start
    (about 20 s, ten times the model's largest value)."""
    path = tmp_path / "origin_only.net"
    path.write_text(origin_only_net(s))
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    roots = _steady_states(sim, seed=0)
    assert len(roots) == 1, roots
    assert np.abs(roots.states).max() <= np.finfo(float).eps * 20 * s
    assert roots.stability == ["stable"]
