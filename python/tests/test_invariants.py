"""Fast invariants: two public ways of computing the same answer must agree (#701).

#689-#699 were silent wrong results that the rest of the suite did not catch,
and ten of the eleven had one shape: two public entry points that should compute
the same thing returned different numbers. None of the checks below needs an
oracle, a golden file or a corpus. Each compares bngsim with itself, on a model
small enough that the whole module runs in a few seconds.

Invariant (equal within solver tolerance), and what it would have caught:

* ``run(0 -> T)`` == ``run_until(T/2)`` then ``run_until(T)`` -- #693
* a ``Simulator`` reused after ``set_param`` == a fresh ``Simulator`` -- #697
* ``run()`` == the matching row of ``run_batch``, with ``squeeze=False``
  (#697) and ``squeeze=True`` (#698)
* ``set_param(p, v)``, ``reset()``, run == reloading the model at ``p = v``
  -- #694, #695, #696
* every parameter is unchanged by a run and repeated runs agree, plain and with
  ``sensitivity_params`` -- #690
* codegen == the interpreter on the same model -- #689, #699
* the trajectory of a sensitivity run == the plain run -- #689
* SSA mean == the exact chemical-master-equation mean -- #692

#691 (the ``jacobian="auto"`` retry starting from mid-run state) needs a model
whose first attempt fails; it is not an invariant of this kind.

The fixture models cover the features where those defects clustered, one
feature per model where that keeps a quarantine tied to one issue. An invariant
is paired only with the fixtures that declare what it needs (a parameter to
write, parameters to differentiate by), so an inapplicable combination is never
generated rather than skipped. A pairing that fails today carries
``xfail(strict=True, raises=...)`` naming its issue in ``Fixture.known``, so the
quarantine retires itself: the fix makes the test XPASS, a strict XPASS fails,
and the fix PR has to delete the entry.

An invariant cannot see a wrong answer that both paths share. The golden
trajectories of #582 catch that kind of drift; #700 and #702 cover the SBML
suites and the nightly oracle sweep.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bngsim
import numpy as np
import pytest

RTOL = 1e-8
ATOL = 1e-10
# Two integrations of the same model at rtol=1e-8 agree far inside this; every
# trajectory defect this module pins is O(1) relative. (#690's parameter drift is
# checked exactly, not through this tolerance.)
CMP_RTOL = 1e-5
CMP_ATOL = 1e-8


# ─── Model text ───────────────────────────────────────────────────────────────
#
# Every model is built by a function of its parameter values, so "reload the
# model with p = v" is the same text with one number changed. That is the
# reference for the set_param invariants, and it involves no bngsim write path.


def _sbml(
    body: str,
    *,
    compartments: str,
    species: str,
    parameters: str,
) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="m">
    <listOfCompartments>{compartments}</listOfCompartments>
    <listOfSpecies>{species}</listOfSpecies>
    <listOfParameters>{parameters}</listOfParameters>
    {body}
  </model>
</sbml>
"""


def _cmp(cid: str, size: float) -> str:
    return f'<compartment id="{cid}" size="{size!r}" spatialDimensions="3" constant="true"/>'


def _spc(
    sid: str,
    comp: str,
    *,
    amount: float | None = None,
    conc: float | None = None,
    hosu: bool = True,
    extra: str = "",
) -> str:
    init = (
        f'initialAmount="{amount!r}"' if amount is not None else f'initialConcentration="{conc!r}"'
    )
    return (
        f'<species id="{sid}" compartment="{comp}" {init} '
        f'hasOnlySubstanceUnits="{str(hosu).lower()}" boundaryCondition="false" '
        f'constant="false"{extra}/>'
    )


def _par(pid: str, value: float, *, constant: bool = True) -> str:
    return f'<parameter id="{pid}" value="{value!r}" constant="{str(constant).lower()}"/>'


def _math(mathml_inner: str) -> str:
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML">{mathml_inner}</math>'


def _rxn(
    rid: str,
    reactants: list[tuple[str, str]],
    products: list[tuple[str, str]],
    kinetic_mathml: str,
) -> str:
    """``reactants`` / ``products`` are ``(species, stoichiometry-attrs)`` pairs."""

    def refs(items: list[tuple[str, str]]) -> str:
        return "".join(
            f'<speciesReference species="{s}" constant="true" {attrs}/>' for s, attrs in items
        )

    rs = f"<listOfReactants>{refs(reactants)}</listOfReactants>" if reactants else ""
    ps = f"<listOfProducts>{refs(products)}</listOfProducts>" if products else ""
    return (
        f'<reaction id="{rid}" reversible="false">{rs}{ps}'
        f"<kineticLaw>{_math(kinetic_mathml)}</kineticLaw></reaction>"
    )


def _times(*xs: str) -> str:
    return "<apply><times/>" + "".join(xs) + "</apply>"


def _ci(x: str) -> str:
    return f"<ci>{x}</ci>"


def _cn(x: float) -> str:
    return f"<cn>{x!r}</cn>"


def _plus(*xs: str) -> str:
    return "<apply><plus/>" + "".join(xs) + "</apply>"


_TIME = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
)


def _event(eid: str, at: float, assignments: dict[str, str], delay: float | None = None) -> str:
    """An ``initialValue=false`` event triggered by ``time >= at``."""
    trigger = _math(f"<apply><geq/>{_TIME}{_cn(at)}</apply>")
    delay_xml = f"<delay>{_math(_cn(delay))}</delay>" if delay is not None else ""
    ea = "".join(
        f'<eventAssignment variable="{v}">{_math(m)}</eventAssignment>'
        for v, m in assignments.items()
    )
    return (
        f'<event id="{eid}" useValuesFromTriggerTime="true">'
        f'<trigger initialValue="false" persistent="true">{trigger}</trigger>{delay_xml}'
        f"<listOfEventAssignments>{ea}</listOfEventAssignments></event>"
    )


# ── SBML: events (#693) ───────────────────────────────────────────────────────
#
# A decays; an initialValue=false event doses it once at t >= 5, and a delayed
# event fires at t >= 9 and executes at t = 11 with the value frozen at 9. The
# leg boundary used by the split-run invariant (t = 10) falls between those two
# times, so a pending execution has to survive it.


def sbml_events(p: Mapping[str, float]) -> bngsim.Model:
    events = (
        "<listOfEvents>"
        + _event(
            "dose",
            5.0,
            {"A": _plus(_ci("A"), _ci("dose_amt")), "n": _plus(_ci("n"), _cn(1.0))},
        )
        + _event("late", 9.0, {"B": _plus(_ci("A"), _cn(5.0))}, delay=2.0)
        + "</listOfEvents>"
    )
    rxns = (
        "<listOfReactions>"
        + _rxn("R1", [("A", 'stoichiometry="1"')], [], _times(_ci("k1"), _ci("A")))
        + _rxn("R2", [("B", 'stoichiometry="1"')], [], _times(_ci("k1"), _ci("B")))
        + "</listOfReactions>"
    )
    return bngsim.Model.from_sbml_string(
        _sbml(
            rxns + events,
            compartments=_cmp("C", 1.0),
            species=_spc("A", "C", amount=10.0) + _spc("B", "C", amount=1.0),
            parameters=_par("k1", p.get("k1", 0.1))
            + _par("dose_amt", p.get("dose_amt", 5.0))
            + _par("n", 0.0, constant=False),
        )
    )


# ── SBML: assignment-rule species + compartment from an initialAssignment ────
#
# X := 2*A is an assignment-rule species (#698 reports it frozen from a squeezed
# batch). The compartment size c is set by c = 2*p (#696 folds it at load). S is
# a concentration species in c and R1's rate k*S is an amount per time, so
# d[S]/dt = -k [S] / c: [S](t) = exp(-k t / (2 p)), and a write to p has to move
# the trajectory.


def sbml_ar_ia(p: Mapping[str, float]) -> bngsim.Model:
    body = f"""
    <listOfInitialAssignments>
      <initialAssignment symbol="c">{_math(_times(_cn(2.0), _ci("p")))}</initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <assignmentRule variable="X">{_math(_times(_cn(2.0), _ci("A")))}</assignmentRule>
    </listOfRules>
    <listOfReactions>
      {_rxn("R1", [("S", 'stoichiometry="1"')], [], _times(_ci("k"), _ci("S")))}
      {_rxn("R2", [("A", 'stoichiometry="1"')], [], _times(_ci("k"), _ci("A")))}
    </listOfReactions>"""
    return bngsim.Model.from_sbml_string(
        _sbml(
            body,
            compartments=_cmp("c", 1.0) + _cmp("C1", 1.0),
            species=_spc("S", "c", conc=1.0, hosu=False)
            + _spc("A", "C1", amount=10.0)
            + _spc("X", "C1", amount=0.0),
            parameters=_par("p", p.get("p", 1.0)) + _par("k", p.get("k", 1.0)),
        )
    )


# ── SBML: non-unit compartment, concentration species, bimolecular (#692/#697)
#
# 2A -> B at rate k*[A]^2*V (an amount per time). ``A0`` is the initial
# concentration, a knob of the builder only, so the SSA test can hold the
# molecule count at 20 while it varies V.


def sbml_dimer(p: Mapping[str, float]) -> bngsim.Model:
    body = f"""
    <listOfReactions>
      {
        _rxn(
            "R1",
            [("A", 'stoichiometry="2"')],
            [("B", 'stoichiometry="1"')],
            _times(_ci("k"), _ci("A"), _ci("A"), _ci("V")),
        )
    }
    </listOfReactions>"""
    return bngsim.Model.from_sbml_string(
        _sbml(
            body,
            compartments=_cmp("V", p.get("V", 10.0)),
            species=_spc("A", "V", conc=p.get("A0", 2.0), hosu=False)
            + _spc("B", "V", conc=0.0, hosu=False),
            parameters=_par("k", p.get("k", 0.05)),
        )
    )


# ── SBML: conversionFactor and parameter-valued stoichiometry (#695) ─────────
#
# P is produced at kp with species conversionFactor cf: P(t) = cf*kp*t. Q is
# produced with stoichiometry sr_P set by the initialAssignment sr_P = z + 1:
# Q(t) = (z + 1)*kq*t. Both folded to numbers at load today.


def sbml_cf_stoich(p: Mapping[str, float]) -> bngsim.Model:
    body = f"""
    <listOfInitialAssignments>
      <initialAssignment symbol="sr_P">{_math(_plus(_ci("z"), _cn(1.0)))}</initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      {_rxn("R1", [], [("P", 'stoichiometry="1"')], _ci("kp"))}
      {_rxn("R2", [], [("Q", 'id="sr_P" stoichiometry="1"')], _ci("kq"))}
    </listOfReactions>"""
    return bngsim.Model.from_sbml_string(
        _sbml(
            body,
            compartments=_cmp("C", 1.0),
            species=_spc("P", "C", amount=0.0, extra=' conversionFactor="cf"')
            + _spc("Q", "C", amount=0.0),
            parameters=_par("cf", p.get("cf", 1.0))
            + _par("z", p.get("z", 1.0))
            + _par("kp", p.get("kp", 0.5))
            + _par("kq", p.get("kq", 1.0)),
        )
    )


# ── .net models ──────────────────────────────────────────────────────────────
#
# One .net feature per model, so that fixing one issue retires exactly one
# quarantine marker.


def _net(params: str, functions: str, species: str, reactions: str, groups: str) -> bngsim.Model:
    text = (
        f"begin parameters\n{params}end parameters\n"
        + (f"begin functions\n{functions}end functions\n" if functions else "")
        + f"begin species\n{species}end species\n"
        + f"begin reactions\n{reactions}end reactions\n"
        + f"begin groups\n{groups}end groups\n"
    )
    return _from_net_text(text)


def net_derived(p: Mapping[str, float]) -> bngsim.Model:
    """``k2 = 2*k1`` is a derived parameter. #694 (an override kept its chain rule
    on every Simulator) retired with the ``.net`` codegen path (#803); a Simulator
    built *before* the override still keeps it, which is #708."""
    k2 = f"{p['k2']!r}  # Constant" if "k2" in p else "2*k1  # ConstantExpression"
    return _net(
        f"    1 k1  {p.get('k1', 0.3)!r}  # Constant\n    2 k2  {k2}\n",
        "",
        "    1 A() 10.0\n    2 B() 0.0\n",
        "    1 1 2 k2 #_R1\n",
        "    1 Atot 1\n    2 Btot 2\n",
    )


def net_sci(p: Mapping[str, float]) -> bngsim.Model:
    """BNG2.pl's scientific-notation rate factor ``2e-06*kp`` (#689: codegen compiled 0
    until #803 compiled .net models from the built model)."""
    return _net(
        f"    1 kp  {p.get('kp', 2.0e5)!r}  # Constant\n",
        "",
        "    1 A() 10.0\n    2 B() 0.0\n",
        "    1 1 2 2e-06*kp #_R1\n",
        "    1 Atot 1\n    2 Btot 2\n",
    )


def net_fwd(p: Mapping[str, float]) -> bngsim.Model:
    """``y()`` references ``x()``, declared after it (#699: emitted in file order until
    #803 compiled .net models from the built model)."""
    return _net(
        f"    1 kx  {p.get('kx', 0.5)!r}  # Constant\n",
        "    1 y() x()*3\n    2 x() kx*Atot/(1+Atot)\n",
        "    1 A() 10.0\n    2 B() 0.0\n",
        "    1 1 2 y #_R1\n",
        "    1 Atot 1\n    2 Btot 2\n",
    )


def net_abs(p: Mapping[str, float]) -> bngsim.Model:
    """``abs()`` in a rate, so the analytic sensitivity RHS is declined (#690)."""
    return _net(
        f"    1 kon  {p.get('kon', 0.1)!r}  # Constant\n"
        f"    2 scale  {p.get('scale', 2.0)!r}  # Constant\n",
        "    1 r() kon*abs(scale)\n",
        "    1 A() 5.0\n    2 B() 0.0\n",
        "    1 1 2 r #_R1\n",
        "    1 Atot 1\n    2 Btot 2\n",
    )


#: .net models are loaded from a file; one directory per session holds them.
_NET_DIR = tempfile.TemporaryDirectory(prefix="bngsim_invariants_")


def _from_net_text(text: str) -> bngsim.Model:
    path = Path(_NET_DIR.name) / (hashlib.sha256(text.encode()).hexdigest()[:16] + ".net")
    path.write_text(text)
    return bngsim.Model.from_net(path)


# ─── Fixture registry ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Fixture:
    """A tiny model plus what the invariants need to know about it.

    ``write`` is the ``(parameter, new value)`` the write invariants apply;
    ``sens`` the sensitivity parameters for the sensitivity invariants. An
    invariant whose inputs a fixture does not declare is never paired with it.
    """

    name: str
    build: Callable[[Mapping[str, float]], bngsim.Model]
    t_end: float
    write: tuple[str, float] | None = None
    sens: tuple[str, ...] = ()
    #: ``(invariant, issue number, raises)`` for every pairing that fails today.
    known: tuple[tuple[str, int, type[BaseException]], ...] = ()


FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        "sbml_events",
        sbml_events,
        t_end=20.0,
        write=("dose_amt", 3.0),
        known=(("split_run", 693, AssertionError),),
    ),
    Fixture(
        "sbml_ar_ia",
        sbml_ar_ia,
        t_end=2.0,
        write=("p", 2.0),
        sens=("p", "k"),
        known=(
            ("run_batch_squeeze", 698, AssertionError),
            ("set_param_vs_reload", 696, AssertionError),
        ),
    ),
    Fixture(
        "sbml_dimer",
        sbml_dimer,
        t_end=5.0,
        write=("V", 4.0),
        sens=("k",),
        known=(
            ("reuse_after_set_param", 697, AssertionError),
            ("run_batch", 697, AssertionError),
        ),
    ),
    Fixture(
        "sbml_cf_stoich",
        sbml_cf_stoich,
        t_end=2.0,
        write=("cf", 3.0),
        sens=("cf", "z", "kp"),
        known=(("set_param_vs_reload", 695, AssertionError),),
    ),
    Fixture(
        "sbml_stoich_ia",
        sbml_cf_stoich,
        t_end=2.0,
        write=("z", 3.0),
        known=(("set_param_vs_reload", 695, AssertionError),),
    ),
    Fixture(
        "net_derived",
        net_derived,
        t_end=1.0,
        write=("k2", 5.0),
        sens=("k1",),
        known=(
            ("reuse_after_set_param", 708, AssertionError),
            ("run_batch", 708, AssertionError),
            ("run_batch_squeeze", 708, AssertionError),
        ),
    ),
    Fixture(
        "net_sci",
        net_sci,
        t_end=2.0,
        write=("kp", 4.0e5),
        sens=("kp",),
    ),
    Fixture(
        "net_fwd",
        net_fwd,
        t_end=2.0,
        write=("kx", 1.5),
    ),
    Fixture(
        "net_abs",
        net_abs,
        t_end=5.0,
        write=("kon", 0.2),
        sens=("scale",),
    ),
)


_NEEDS_WRITE = {"reuse_after_set_param", "run_batch", "run_batch_squeeze", "set_param_vs_reload"}
_NEEDS_SENS = {"params_unchanged_sens", "sens_traj_vs_plain"}


def _applies(inv: str, fx: Fixture) -> bool:
    """Whether ``fx`` declares what invariant ``inv`` needs; if not, no case exists."""
    if inv in _NEEDS_WRITE and fx.write is None:
        return False
    return not (inv in _NEEDS_SENS and not fx.sens)


def _cases(inv: str) -> list[Any]:
    out = []
    for fx in FIXTURES:
        if not _applies(inv, fx):
            continue
        marks = [
            pytest.mark.xfail(strict=True, raises=raises, reason=f"#{issue}")
            for name, issue, raises in fx.known
            if name == inv
        ]
        out.append(pytest.param(fx, id=fx.name, marks=marks))
    return out


# ─── Comparison helpers ──────────────────────────────────────────────────────


def _sim(model: bngsim.Model, **kw: Any) -> bngsim.Simulator:
    return bngsim.Simulator(model, method="ode", **kw)


def _run(sim: bngsim.Simulator, fx: Fixture, **kw: Any) -> bngsim.Result:
    return sim.run(t_span=(0.0, fx.t_end), n_points=11, rtol=RTOL, atol=ATOL, **kw)


def _report(r: Any) -> dict[str, np.ndarray]:
    """Every reported column of a 2-D Result, keyed by where it came from.

    Amounts (``as_roadrunner`` without brackets) are included because #697 was
    wrong only there: the stored concentrations were right.
    """
    out: dict[str, np.ndarray] = {"time": np.asarray(r.time)}
    for i, n in enumerate(r.species_names):
        out[f"[{n}]"] = np.asarray(r.species)[:, i]
    amounts = r.as_roadrunner(list(r.species_names))
    for i, n in enumerate(r.species_names):
        out[n] = np.asarray(amounts)[:, i]
    for n in getattr(r, "observable_names", []) or []:
        out[f"obs:{n}"] = np.asarray(r.observables[n])
    for n in getattr(r, "expression_names", []) or []:
        out[f"expr:{n}"] = np.asarray(r.expressions[n])
    sens = np.asarray(r.sensitivities)
    if sens.size:
        for j, p in enumerate(r.sensitivity_params):
            for i, n in enumerate(r.species_names):
                out[f"d[{n}]/d{p}"] = sens[:, i, j]
    return out


def _assert_same(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], what: str) -> None:
    assert sorted(a) == sorted(b), f"{what}: different columns {sorted(a)} vs {sorted(b)}"
    bad = []
    for k in a:
        if not np.allclose(a[k], b[k], rtol=CMP_RTOL, atol=CMP_ATOL, equal_nan=True):
            diff = np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k])))
            bad.append(
                f"{k}: max|diff|={diff:.3g}\n      {np.asarray(a[k])}\n   vs {np.asarray(b[k])}"
            )
    assert not bad, f"{what} disagree on {len(bad)} column(s):\n  " + "\n  ".join(bad)


def _params(model: bngsim.Model) -> dict[str, float]:
    """Every parameter value; .net functions share the namespace but are outputs."""
    functions = set(model.function_names)
    return {n: model.get_param(n) for n in model.param_names if n not in functions}


# ─── The invariants ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("fx", _cases("split_run"))
def test_split_run(fx: Fixture) -> None:
    """``run(0 -> T)`` equals ``run_until(T/2)`` followed by ``run_until(T)`` (#693)."""
    whole = _run(_sim(fx.build({})), fx)
    sim = _sim(fx.build({}))
    t_mid = fx.t_end / 2
    first = sim.run_until(t_mid, n_points=6, rtol=RTOL, atol=ATOL)
    second = sim.run_until(fx.t_end, n_points=6, rtol=RTOL, atol=ATOL)
    joined = {k: np.concatenate([v, _report(second)[k][1:]]) for k, v in _report(first).items()}
    _assert_same(_report(whole), joined, "one run and two run_until legs")


@pytest.mark.parametrize("fx", _cases("reuse_after_set_param"))
def test_reuse_after_set_param(fx: Fixture) -> None:
    """A ``Simulator`` reused after ``set_param`` equals a fresh one (#697)."""
    p, v = fx.write
    model = fx.build({})
    sim = _sim(model, sensitivity_params=list(fx.sens) or None)
    _run(sim, fx)
    model.set_param(p, v)
    model.reset()
    reused = _report(_run(sim, fx))

    fresh_model = fx.build({})
    fresh_model.set_param(p, v)
    fresh_model.reset()
    fresh = _report(_run(_sim(fresh_model, sensitivity_params=list(fx.sens) or None), fx))
    _assert_same(fresh, reused, f"fresh and reused Simulator after set_param({p!r}, {v})")


def _batch_vs_run(fx: Fixture, squeeze: bool) -> None:
    p, v = fx.write
    sens = list(fx.sens) or None
    sim = _sim(fx.build({}), sensitivity_params=sens)
    points = [{p: fx.build({}).get_param(p)}, {p: v}]
    batch = sim.run_batch(
        t_span=(0.0, fx.t_end), n_points=11, params=points, rtol=RTOL, atol=ATOL, squeeze=squeeze
    )
    for i, point in enumerate(points):
        model = fx.build({})
        model.set_params(point)
        model.reset()
        single = _report(_run(_sim(model, sensitivity_params=sens), fx))
        if squeeze:
            row = {"time": np.asarray(batch.time)}
            for j, n in enumerate(batch.species_names):
                row[f"[{n}]"] = np.asarray(batch.species)[i, :, j]
            s = np.asarray(batch.sensitivities)
            if s.size:
                for k, q in enumerate(batch.sensitivity_params):
                    for j, n in enumerate(batch.species_names):
                        row[f"d[{n}]/d{q}"] = s[i, :, j, k]
            single = {k: single[k] for k in row}
        else:
            row = _report(batch[i])
        _assert_same(single, row, f"run() and run_batch(squeeze={squeeze}) row {i} ({point})")


@pytest.mark.parametrize("fx", _cases("run_batch"))
def test_run_equals_run_batch_row(fx: Fixture) -> None:
    """``run()`` equals the matching row of ``run_batch(squeeze=False)`` (#697)."""
    _batch_vs_run(fx, squeeze=False)


@pytest.mark.parametrize("fx", _cases("run_batch_squeeze"))
def test_run_equals_squeezed_run_batch_row(fx: Fixture) -> None:
    """``run()`` equals the matching row of ``run_batch(squeeze=True)`` (#698)."""
    _batch_vs_run(fx, squeeze=True)


@pytest.mark.parametrize("fx", _cases("set_param_vs_reload"))
def test_set_param_equals_reload(fx: Fixture) -> None:
    """``set_param(p, v)``, ``reset()``, run equals reloading with ``p = v`` (#694-#696)."""
    p, v = fx.write
    sens = list(fx.sens) or None
    written = fx.build({})
    written.set_param(p, v)
    written.reset()
    a = _report(_run(_sim(written, sensitivity_params=sens), fx))
    b = _report(_run(_sim(fx.build({p: v}), sensitivity_params=sens), fx))
    _assert_same(b, a, f"reload at {p}={v} and set_param({p!r}, {v})")


def _params_unchanged(fx: Fixture, sens: list[str] | None) -> None:
    model = fx.build({})
    before = _params(model)
    sim = _sim(model, sensitivity_params=sens)
    first = _report(_run(sim, fx))
    assert _params(model) == pytest.approx(before, rel=1e-14, abs=0), "a run changed a parameter"
    for _ in range(3):
        model.reset()
        again = _report(_run(sim, fx))
    assert _params(model) == pytest.approx(before, rel=1e-14, abs=0), "repeated runs drift"
    _assert_same(first, again, "the first and fourth identical run")


@pytest.mark.parametrize("fx", _cases("params_unchanged"))
def test_run_leaves_params_unchanged(fx: Fixture) -> None:
    """A plain run leaves every parameter as it found it, and repeats exactly."""
    _params_unchanged(fx, None)


@pytest.mark.parametrize("fx", _cases("params_unchanged_sens"))
def test_sensitivity_run_leaves_params_unchanged(fx: Fixture) -> None:
    """So does a run with ``sensitivity_params`` (#690)."""
    _params_unchanged(fx, list(fx.sens))


@pytest.mark.parametrize("fx", _cases("codegen_vs_interpreter"))
def test_codegen_equals_interpreter(fx: Fixture) -> None:
    """Compiled RHS equals the interpreter on the same model (#689, #699)."""
    interp = _report(_run(_sim(fx.build({}), codegen=False), fx))
    compiled_sim = _sim(fx.build({}), codegen=True)
    compiled = _report(_run(compiled_sim, fx))
    _assert_same(interp, compiled, "interpreter and codegen")


@pytest.mark.parametrize("fx", _cases("sens_traj_vs_plain"))
def test_sensitivity_run_trajectory_equals_plain(fx: Fixture) -> None:
    """Asking for sensitivities does not change the trajectory (#689)."""
    plain = _report(_run(_sim(fx.build({})), fx))
    with_sens = _report(_run(_sim(fx.build({}), sensitivity_params=list(fx.sens)), fx))
    _assert_same(plain, {k: with_sens[k] for k in plain}, "plain and sensitivity run")


# ── SSA against the exact chemical master equation (#692) ───────────────────
#
# 2A -> B with n0 = 20 molecules. The exact propensity is k*n*(n-1)/V; k is
# scaled with V so every volume has the same exact kinetics (and one CME), with
# the mean count near half of n0 at the sampled time. The state space is
# n0/2 + 1 states and the CME is solved exactly by a matrix exponential. At
# V = 1 the defect is invisible (count and concentration coincide); at V = 10 it
# is not.

_N0 = 20
_C = 0.005  # k / V: the exact per-pair propensity constant
_T_SSA = 5.0
_N_REP = 400


def _cme_mean_count(t: float) -> float:
    from scipy.linalg import expm

    states = np.arange(_N0, -1, -2)  # n = 20, 18, ..., 0
    Q = np.zeros((len(states), len(states)))
    for i, n in enumerate(states[:-1]):
        a = _C * n * (n - 1)
        Q[i, i] -= a
        Q[i + 1, i] += a
    p0 = np.zeros(len(states))
    p0[0] = 1.0
    return float(states @ (expm(Q * t) @ p0))


@pytest.mark.parametrize("V", [pytest.param(1.0, id="V=1"), pytest.param(10.0, id="V=10")])
def test_ssa_mean_equals_cme_mean(V: float) -> None:
    """The SSA mean count is within 5 standard errors of the exact CME mean (#692)."""
    model = sbml_dimer({"V": V, "A0": _N0 / V, "k": _C * V})
    sim = bngsim.Simulator(model, method="ssa")
    runs = sim.run_replicates(_N_REP, t_span=(0.0, _T_SSA), n_points=2, seed=701)
    i = list(runs[0].species_names).index("A")
    counts = np.array([np.asarray(r.species)[-1, i] * V for r in runs])
    exact = _cme_mean_count(_T_SSA)
    se = counts.std(ddof=1) / math.sqrt(_N_REP)
    z = (counts.mean() - exact) / max(se, 1e-12)
    assert abs(z) < 5.0, f"SSA mean A count {counts.mean():.4f} vs CME {exact:.4f} ({z:+.1f} SE)"
