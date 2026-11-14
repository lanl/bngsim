"""A `Sat` or `Hill` rate law on a repeated reactant loads (issue #1527).

BNG2.pl writes `A() + A() -> B() Sat(kcat,Km)` as `1 1,1 2 0.5*Sat kcat Km`, and
`Hill(kcat,Km,h)` as `0.5*Hill kcat Km h`, and run_network simulates both. The
loader rewrites a Sat or Hill law as a Functional one, and it refused any
reactant that appeared more than once. The rewrite is exact there if it reads
reactant positions, as Network3's `rxn_rate` does: Sat is
`stat*k*prod_{i<nK} x_i/(K_i+x_i)*prod_{i>=nK} x_i` and Hill is
`stat*V*x0^h/(K^h+x0^h)*prod_{i>=1} x_i`, and a Functional reaction multiplies its
function by every reactant position (the falling factorial for SSA).

What this file pins:

* the lines BNG2.pl 2.9.3 writes for a homodimer load, from both doors, and give
  run_network's A(1);
* the rewrite is the same model as the explicit function written over the same
  reactant positions, for the ODE (interpreted and compiled), SSA and the
  sensitivities, including a second saturation constant on the repeated
  reactant, which BNG2.pl counts against the reactant positions.
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._net_reader import build_model_from_parsed, parse_net_file


def _has_cc() -> bool:
    from bngsim import _codegen as cg

    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")

#: A(1) from run_network (BioNetGen 2.9.3) on the .net BNG2.pl writes for
#: `A() + A() -> B() <law>` with kcat = 1.2, Km = 3, h = 2, A(0) = 10,
#: `simulate({method=>"ode",t_end=>1,n_steps=>1,atol=>1e-12,rtol=>1e-12})`.
#: Each is also the closed form: Sat solves -Km/A + ln A = const - kcat*t and
#: Hill -Km^h/(h*A^h) + ln A = const - kcat*t.
RUN_NETWORK_A1 = {"Sat": 4.407330663, "Hill": 3.881617517}

#: The law as BNG2.pl writes it, and the same rate as an explicit function over
#: the same reactant positions, which the loader already read.
LAWS = {
    "Sat": ("0.5*Sat kcat Km", "kcat/(Km+At)"),
    "Hill": ("0.5*Hill kcat Km h", "kcat*At^(h-1)/(Km^h+At^h)"),
    # BNG2.pl writes `A() + A() -> B() Sat(kcat,Km,K2)` so: one saturation
    # constant per reactant position, which it and run_network count against
    # the two positions of A rather than the one species.
    "Sat2": ("0.5*Sat kcat Km K2", "kcat/((Km+At)*(K2+At))"),
}


def _net(tmp_path: Path, name: str, rate: str, functions: str = "") -> Path:
    p = tmp_path / f"{name}.net"
    p.write_text(
        textwrap.dedent(
            f"""
            # Created by BioNetGen 2.9.3
            begin parameters
                1 kcat 1.2
                2 Km   3.0
                3 h    2.0
                4 K2   5.0
            end parameters
            begin species
                1 A() 10
                2 B() 0
            end species
            begin reactions
                1 1,1 2 {rate} #_R1
            end reactions
            begin groups
                1 At 1
            end groups
            """
        ).strip()
        + "\n"
        + functions
    )
    return p


def _quiet(fn, *args):
    # Sat and Hill are rewritten as functions, with a UserWarning that says so.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*args)


def _pair(tmp_path: Path, kind: str):
    """The model as BNG2.pl writes it, and the explicit-function control."""
    law, body = LAWS[kind]
    written = _quiet(bngsim.Model.from_net, str(_net(tmp_path, kind, law)))
    control = bngsim.Model.from_net(
        str(
            _net(
                tmp_path,
                kind + "_fn",
                "0.5*f",
                f"begin functions\n    1 f() {body}\nend functions\n",
            )
        )
    )
    return written, control


@pytest.mark.parametrize("kind", ["Sat", "Hill"])
@pytest.mark.parametrize("codegen", [False, pytest.param(True, marks=requires_cc)])
def test_both_doors_give_run_networks_value(tmp_path: Path, kind: str, codegen: bool) -> None:
    net = _net(tmp_path, kind, LAWS[kind][0])
    parsed = _quiet(parse_net_file, net)
    (rxn,) = parsed["reactions"]
    assert (rxn["type"], rxn["stat_factor"]) == ("functional", 0.5)
    loaded = _quiet(bngsim.Model.from_net, str(net))
    assert build_model_from_parsed(parsed)._core.codegen_data() == loaded._core.codegen_data()
    run = bngsim.Simulator(loaded, method="ode", codegen=codegen).run(
        sample_times=[0.0, 1.0], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(run.observables["At"])[-1], RUN_NETWORK_A1[kind], rtol=1e-8
    )


@pytest.mark.parametrize("kind", list(LAWS))
@pytest.mark.parametrize("codegen", [False, pytest.param(True, marks=requires_cc)])
def test_ode_is_the_explicit_function(tmp_path: Path, kind: str, codegen: bool) -> None:
    runs = [
        bngsim.Simulator(m, method="ode", codegen=codegen).run(
            t_span=(0.0, 5.0), n_points=11, rtol=1e-12, atol=1e-12
        )
        for m in _pair(tmp_path, kind)
    ]
    np.testing.assert_allclose(
        np.asarray(runs[0].species), np.asarray(runs[1].species), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("kind", list(LAWS))
def test_ssa_is_the_explicit_function(tmp_path: Path, kind: str) -> None:
    """A(A-1) for the two positions of A, as Network3's (x-n) numerators read it."""
    runs = [
        bngsim.Simulator(m, method="ssa").run((0.0, 5.0), 11, seed=11)
        for m in _pair(tmp_path, kind)
    ]
    np.testing.assert_array_equal(np.asarray(runs[0].species), np.asarray(runs[1].species))


@requires_cc
@pytest.mark.parametrize("kind", list(LAWS))
def test_sensitivities_are_the_explicit_functions(tmp_path: Path, kind: str) -> None:
    params = ["kcat", "Km"] + (["h"] if kind == "Hill" else ["K2"] if kind == "Sat2" else [])
    sims = [
        bngsim.Simulator(m, method="ode", sensitivity_params=params) for m in _pair(tmp_path, kind)
    ]
    assert all(sim.has_analytic_sens_rhs for sim in sims)
    got, want = (
        np.asarray(sim.run(t_span=(0.0, 5.0), n_points=11, rtol=1e-12, atol=1e-13).sensitivities)
        for sim in sims
    )
    assert np.all(np.isfinite(got)) and np.any(got != 0.0)
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)
