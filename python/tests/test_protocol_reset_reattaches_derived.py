"""resetParameters re-attaches every derived parameter overridden since the save (issue #1870).

BNG2.pl's ``resetParameters`` puts back the parameter list ``saveParameters``
copied, expressions included: a derived parameter that a ``setParameter``
made a number in between follows what it reads again. bngsim re-attaches one
by writing back the value its expression gives (issue #188), which works only
once the parameters it reads are back too. The replay wrote them back in the
order of a ``set``, so a parameter written before one it reads stayed a
number, at the saved value, and stopped following: wrong results with no
error, in some processes and not in others (the order follows the string
hash seed).

Oracle: the closed form for ``A <-> B`` at ``kf``, ``kr`` from ``A(0) = 100``,
at the rates the network's expressions give once the reset has put them back.
BNG2.pl 2.9.3 on the issue's file restores both expressions (the ``.net`` it
writes after the reset lists ``k2 k1*0.5`` and ``k3 k2*2``).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from itertools import permutations
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim.convert._protocol import Experiment, ProtocolSpec, StateChange

# Tight enough that the solver's error is far below the gap between a right
# answer and a wrong one (percent), loose enough for any platform's libm.
TOL = {"rtol": 1e-10, "atol": 1e-12}
REL = 1e-8

# A <-> B at the last two parameters of a chain, each reading the one before.
NET = """begin parameters
{parameters}
end parameters
begin species
    1 A() 100
    2 B() 0
end species
begin reactions
    1 1 2 {kf}
    2 2 1 {kr}
end reactions
begin groups
    1 Atot 1
end groups
"""


def a_at(t: float, kf: float, kr: float) -> float:
    # dA/dt = -kf A + kr (100 - A), A(0) = 100.
    return 100 * (kr + kf * math.exp(-(kf + kr) * t)) / (kf + kr)


def _model(tmp_path: Path, declared: list[tuple[str, str]], kf: str, kr: str) -> bngsim.Model:
    lines = "\n".join(f"    {i} {n} {e}" for i, (n, e) in enumerate(declared, 1))
    net = tmp_path / "m.net"
    net.write_text(NET.format(parameters=lines, kf=kf, kr=kr), encoding="utf-8")
    return bngsim.Model.from_net(str(net))


def _set(name: str, value: float) -> StateChange:
    return StateChange(kind="set_parameter", target=name, value=float(value))


SAVE = StateChange(kind="save_parameters")
RESET = StateChange(kind="reset_parameters")
RUN = Experiment(kind="simulate", method="ode", t_span=(0.0, 0.25), n_points=2)


def _replay(model, *steps):
    return bngsim.run_protocol(model, ProtocolSpec(steps=tuple(steps)), **TOL)


def _end(run) -> float:
    return float(run.result.species[-1, 0])


def _live(model) -> dict[str, bool]:
    return dict(zip(model.param_names, map(bool, model.param_is_expression), strict=True))


# k2 = k1/2 and k3 = 2 k2: 0.5 and 1 as declared, 1.5 and 3 at k1 = 3.
PAIR = [("k1", "1"), ("k2", "k1*0.5"), ("k3", "k2*2")]


@pytest.mark.parametrize("declared", [PAIR, [PAIR[0], PAIR[2], PAIR[1]]], ids=["k2-k3", "k3-k2"])
@pytest.mark.parametrize("first", ["k2", "k3"])
def test_two_derived_parameters_overridden_then_reset_both_follow_again(tmp_path, declared, first):
    # The issue's protocol, the network's two lines in either order and the
    # two sets in either order. After the reset, k1 = 3 moves both.
    model = _model(tmp_path, declared, "k2", "k3")
    sets = [_set("k2", 0.7), _set("k3", 5)]
    out = _replay(model, SAVE, *(sets if first == "k2" else sets[::-1]), RESET, _set("k1", 3), RUN)
    assert _live(out.model) == {"k1": False, "k2": True, "k3": True}
    assert [out.model.get_param(n) for n in ("k2", "k3")] == [1.5, 3.0]
    assert _end(out[0]) == pytest.approx(a_at(0.25, 1.5, 3.0), rel=REL)
    # Not with k3 left a number, at the saved 1 (the issue's wrong result).
    assert _end(out[0]) != pytest.approx(a_at(0.25, 1.5, 1.0), rel=1e-3)


def test_the_order_a_set_gives_the_two_names_does_not_decide_it(tmp_path):
    # The replay wrote the overridden parameters back in the order a set of
    # their names iterates in, which the string hash seed decides. Here the
    # name that order puts first is made the one that reads the other, so the
    # old code fails in every process, not in some.
    names = ("ka", "kb")
    reader = next(iter({n for n in names}))  # as _param_state builds its set
    (read,) = set(names) - {reader}
    declared = [("k1", "1")] + [(n, "k1*0.5" if n == read else f"{read}*2") for n in names]
    model = _model(tmp_path, declared, read, reader)
    out = _replay(model, SAVE, _set(read, 0.7), _set(reader, 5), RESET, _set("k1", 3), RUN)
    assert _live(out.model) == {"k1": False, "ka": True, "kb": True}
    assert _end(out[0]) == pytest.approx(a_at(0.25, 1.5, 3.0), rel=REL)


# k2 .. k6, each reading earlier ones: 0.5, 1, 1.5, 0.75, 1.75 as declared,
# and three times that at k1 = 3 (every expression is linear in k1).
CHAIN = [
    ("k2", "k1*0.5"),
    ("k3", "k2*2"),
    ("k4", "k3+k2"),
    ("k5", "k4*0.5"),
    ("k6", "k5+k3"),
]
ORDERS = {
    "as-read": CHAIN,
    "reversed": CHAIN[::-1],
    "mixed": [CHAIN[i] for i in (3, 0, 4, 2, 1)],
}


@pytest.mark.parametrize("derived", list(ORDERS.values()), ids=list(ORDERS))
def test_a_chain_of_five_follows_again_whatever_order_the_network_lists_it_in(tmp_path, derived):
    # Every one is overridden after the save. Listed in reverse, each is
    # written back before the one it reads, and only k2 re-attaches on the
    # first pass; by then all five hold their saved values, so the second
    # re-attaches the rest.
    model = _model(tmp_path, [("k1", "1"), *derived], "k5", "k6")
    sets = [_set(n, 7 + i) for i, (n, _) in enumerate(CHAIN)]
    out = _replay(model, SAVE, *sets, RESET, _set("k1", 3), RUN)
    assert all(_live(out.model)[n] for n, _ in CHAIN)
    values = [out.model.get_param(n) for n, _ in CHAIN]
    assert values == [1.5, 3.0, 4.5, 2.25, 5.25]
    assert _end(out[0]) == pytest.approx(a_at(0.25, 2.25, 5.25), rel=REL)


@pytest.mark.parametrize("order", list(permutations(range(3))), ids=lambda o: "".join(map(str, o)))
def test_a_reset_to_a_label_reattaches_in_every_order_of_the_sets(tmp_path, order):
    # saveParameters("p") and resetParameters("p") go the same way, and the
    # order the three were overridden in is no part of it.
    model = _model(tmp_path, [("k1", "1"), *CHAIN[:3]], "k3", "k4")
    save = StateChange(kind="save_parameters", label="p")
    reset = StateChange(kind="reset_parameters", label="p")
    sets = [_set("k2", 0.7), _set("k3", 5), _set("k4", 9)]
    out = _replay(model, save, *(sets[i] for i in order), reset, _set("k1", 3), RUN)
    assert [out.model.get_param(n) for n in ("k2", "k3", "k4")] == [1.5, 3.0, 4.5]
    assert _end(out[0]) == pytest.approx(a_at(0.25, 3.0, 4.5), rel=REL)


def test_a_parameter_overridden_before_the_save_stays_a_number(tmp_path):
    # Only what was an expression at the save is re-attached: k2 was a number
    # then, and the reset puts that number back; k3, overridden after it,
    # follows k2 again.
    model = _model(tmp_path, PAIR, "k2", "k3")
    out = _replay(
        model, _set("k2", 0.7), SAVE, _set("k2", 0.9), _set("k3", 5), RESET, _set("k1", 3), RUN
    )
    assert _live(out.model) == {"k1": False, "k2": False, "k3": True}
    assert [out.model.get_param(n) for n in ("k2", "k3")] == [0.7, 1.4]
    assert _end(out[0]) == pytest.approx(a_at(0.25, 0.7, 1.4), rel=REL)


def test_a_derived_parameter_that_is_nan_does_not_keep_the_reset_going(tmp_path):
    # k2 k1*0/0 is NaN, and NaN equals nothing: writing it back never
    # re-attaches k2. The reset ends all the same, and k3, overridden as well,
    # follows k1 again.
    model = _model(tmp_path, [("k1", "1"), ("k2", "k1*0/0"), ("k3", "k1*2")], "k1", "k3")
    out = _replay(model, SAVE, _set("k2", 4), _set("k3", 5), RESET, _set("k1", 3), RUN)
    assert math.isnan(out.model.get_param("k2"))
    assert _live(out.model)["k3"] and out.model.get_param("k3") == 6.0
    assert _end(out[0]) == pytest.approx(a_at(0.25, 3.0, 6.0), rel=REL)


# --- BNG2.pl itself, where it is installed -----------------------------------

BNGPATH = os.environ.get("BNGPATH")
BNG2 = (Path(BNGPATH).resolve() / "BNG2.pl") if BNGPATH else None
HAS_BNG2 = BNG2 is not None and BNG2.is_file() and bool(shutil.which("perl"))

# The issue's file.
BNGL = """begin model
begin parameters
  k1 1
  k2 k1*0.5
  k3 k2*2
  k4 2
end parameters
begin seed species
  A() 100
  B() 0
  C() 0
end seed species
begin observables
  Molecules Atot A()
  Molecules Ctot C()
end observables
begin reaction rules
  A() <-> B() k1, k2
  B() <-> C() k3, k4
end reaction rules
end model
generate_network({overwrite=>1})
saveParameters()
setParameter("k2",0.7)
setParameter("k3",5)
resetParameters()
setParameter("k1",3)
simulate({method=>"ode",t_end=>0.5,n_steps=>1,atol=>1e-12,rtol=>1e-10})
"""


@pytest.mark.skipif(not HAS_BNG2, reason="needs BNG2.pl on BNGPATH")
def test_run_protocol_agrees_with_bng2pl_on_the_issues_file(tmp_path):
    bngl = tmp_path / "m.bngl"
    bngl.write_text(BNGL, encoding="utf-8")
    subprocess.run(
        ["perl", str(BNG2), "m.bngl"],
        cwd=tmp_path,
        # BNG2.pl finds run_network through BNGPATH, from that directory too.
        env={**os.environ, "BNGPATH": str(BNG2.parent)},
        check=True,
        capture_output=True,
        timeout=300,
    )
    # time, Atot, Ctot at t = 0.5: BNG2.pl 2.9.3 gives 34.62886121754 and
    # 29.16669451203; with k3 left a number, at 1, Ctot is 12.31.
    want = np.atleast_2d(np.loadtxt(tmp_path / "m.gdat"))[-1, 1:3]
    assert want.tolist() == pytest.approx([34.62886121754, 29.16669451203], rel=1e-9)
    model = bngsim.Model.from_bngl(bngl)
    (run,) = bngsim.run_protocol(model, bngsim.convert.parse_bngl_protocol(bngl), **TOL)
    names = list(run.result.observable_names)
    got = [float(np.asarray(run.result.observables)[-1, names.index(n)]) for n in ("Atot", "Ctot")]
    # The solvers differ (BNG2.pl's run_network, bngsim's CVODE), each at
    # rtol 1e-10: seven digits is well inside both, and far from the 58% gap.
    assert got == pytest.approx(want.tolist(), rel=1e-7)
