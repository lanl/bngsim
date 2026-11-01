"""A rate-rule target a hair below an exact zero, on the non-finite retry (issue #1340).

MODEL1812040005's P is a rate-rule target at an exact equilibrium of 0, read by
``a_hat = a*P^z`` with z fractional. Summed in the order its species are
written, its derivative is exactly 0 and P stays 0; reversed, the same sums
leave -3.6e-27, where ``P^z`` is NaN, and the run failed
(``CV_REPTD_RHSFUNC_ERR``). The value RHS's retry on a clamped state left a
rate-rule target alone (#706: it may be negative by design, a membrane potential
at -61). Where the clamp alone leaves the reading non-finite, a slot it leaves
alone that is negative by less than its own atol is now read as 0 on a second
reading, as the sensitivity RHS reads one (GH #392); one negative by more is
left alone, and still fails where its rate law is undefined. A slot the model
takes on down through that band is refused where it leaves it, and the
steady-state run test reads the band as the time course does.

Since #1420 a scalar ``atol`` is a ceiling per species, and a slot that starts
at -5e-10 is held to about rtol times that, so it is past its own band. The
cases here that hold a slot at a value below 0 take ``bngsim.ExactAtol``, which
gives every slot the one number.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import bngsim
import numpy as np
import pytest

_RUN = {"t_span": (0, 10), "n_points": 11, "rtol": 1e-8, "atol": 1e-9}
_EXACT = {**_RUN, "atol": bngsim.ExactAtol(1e-9)}

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
_needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")
_needs_klu = pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled")


def _sinking(c: str, order: str, power: str = "0.5") -> bngsim.Model:
    """P' = -c from P = 0, read by P^power in A's decay: for c = 1e-30, P sinks
    to -1e-29 by t = 10, inside atol; for c = 1e-3, to -1e-2, outside it."""
    decl = "A = 1, P = 0" if order == "AP" else "P = 0, A = 1"
    return bngsim.Model.from_antimony_string(
        f"model m\n  species {decl};\n  k = 0.1; c = {c};\n  P' = -c;\n"
        f"  J: A -> ; k*A*P^{power};\nend"
    )


def _held(p: str) -> bngsim.Model:
    """P held at p by a rate rule (P' = 0), read by sqrt(P) in A's decay."""
    return bngsim.Model.from_antimony_string(
        f"model m\n  species A = 1, P = {p};\n  k = 0.1;\n  P' = 0;\n  J: A -> ; k*A*P^0.5;\nend"
    )


def _run(model, run=_RUN, **simulator):
    r = bngsim.Simulator(model, method="ode", **simulator).run(**run)
    names = list(r.species_names)
    return {n: np.asarray(r.species)[:, names.index(n)] for n in names}


@pytest.mark.parametrize("order", ["AP", "PA"])
def test_a_rate_rule_target_inside_atol_of_zero_is_read_as_zero(order):
    x = _run(_sinking("1e-30", order))
    assert x["P"][-1] == pytest.approx(-1e-29)  # the state itself is not moved
    np.testing.assert_array_equal(x["A"], 1.0)  # sqrt(P) read at P = 0


def test_a_run_with_sensitivities_reads_it_the_same_way():
    """A run with sensitivity columns sets its solver up on another path than a
    plain one (create_cvode_core, not the warm cache), which fills the
    tolerance the retry reads as well. P^1.5, flat at 0: sqrt's slope there is
    infinite, which no reading of P makes a sensitivity of."""
    model = _sinking("1e-30", "AP", power="1.5")
    r = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(**_RUN)
    names = list(r.species_names)
    np.testing.assert_array_equal(np.asarray(r.species)[:, names.index("A")], 1.0)


@pytest.mark.parametrize(
    "codegen", [False, pytest.param(True, marks=_needs_cc)], ids=["interpreted", "compiled"]
)
@pytest.mark.parametrize(
    "sparse", [False, pytest.param(True, marks=_needs_klu)], ids=["dense", "sparse"]
)
def test_the_closed_form_jacobian_reads_the_band_too(sparse, codegen):
    """P^1.5 in A's decay: the Jacobian's 1.5*k*A*P^0.5 has no value for P
    below 0, and is 0 at P = 0. Under jacobian="analytical" no difference
    quotient stands in, so each of the four fills (dense and sparse,
    interpreted and compiled) reads the band itself or the run fails. Through
    P^0.5, whose slope at 0 is infinite under any reading, the cases above fall
    back to the quotient and reach none of them. Codegen emits its sparse fill
    only for a model past the sparse gate, so the sparse rows carry a chain of
    60 species beside, which the band does not touch."""
    model = _sinking("1e-30", "AP", power="1.5")
    if sparse:
        chain = "\n".join(f"  R{i}: X{i} -> X{i + 1}; kx*X{i};" for i in range(1, 60))
        model = bngsim.Model.from_antimony_string(
            "model m\n  species A = 1, P = 0, "
            + ", ".join(f"X{i} = 1" for i in range(1, 61))
            + ";\n  k = 0.1; c = 1e-30; kx = 0.01;\n  P' = -c;\n  J: A -> ; k*A*P^1.5;\n"
            + chain
            + "\nend"
        )
    sim = bngsim.Simulator(
        model,
        method="ode",
        jacobian="analytical",
        force_sparse_linear_solver=sparse,
        codegen=codegen,
    )
    r = sim.run(**_RUN)
    names = list(r.species_names)
    np.testing.assert_array_equal(np.asarray(r.species)[:, names.index("A")], 1.0)


def test_the_answer_does_not_depend_on_the_order_species_are_written():
    a, b = _run(_sinking("1e-30", "AP")), _run(_sinking("1e-30", "PA"))
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


@pytest.mark.parametrize("order", ["AP", "PA"])
def test_a_rate_rule_target_negative_past_atol_is_left_alone(order):
    """The control: past atol it is not numerically zero, and sqrt of it is not
    read as anything. The run fails as it did."""
    with pytest.raises(bngsim.SimulationError):
        _run(_sinking("1e-3", order))


@pytest.mark.parametrize("jacobian", ["auto", "fd"])
def test_a_slot_negative_by_design_keeps_its_value_where_the_clamp_alone_clears_f(jacobian):
    """S is drained past 0 by a zero-order sink and read by S^0.5; it is
    masked, and its clamp alone makes f finite. V is a rate-rule target held at
    -5e-10 (V' = 0), inside its atol of 1e-9, and B grows at -g*V = 0.5, so
    exactly B = 0.5*t. Read in the same pass as S's clamp, V was read at 0 on
    every retry after S passed 0, and B stopped at 0.5 from t = 1 on."""
    model = bngsim.Model.from_antimony_string(
        "model m\n  species S = 1, A = 1, B = 0;\n  V = -5e-10; V' = 0;\n"
        "  c = 1; k = 0.1; g = 1e9;\n  J1: S -> ; c;\n  J2: A -> ; k*A*S^0.5;\n"
        "  J3: -> B; -g*V;\nend"
    )
    r = bngsim.Simulator(model, method="ode", jacobian=jacobian).run(**_EXACT)
    names = list(r.species_names)
    b = np.asarray(r.species)[:, names.index("B")]
    # B' is the constant 0.5, which BDF integrates to rounding; 1e-6 is far
    # inside the gap to the wrong answer (0.5 for 5 at t = 10).
    np.testing.assert_allclose(b, 0.5 * np.asarray(r.time), rtol=1e-6, atol=1e-12)
    assert np.asarray(r.species)[-1, names.index("V")] == -5e-10


@pytest.mark.parametrize(
    ("c", "atol", "t_end"),
    [("1e-10", 1e-12, 1.0), ("1e-10", 1e-9, 20.0), ("1e-6", 1e-9, 10.0)],
)
def test_a_slot_drained_through_its_band_is_refused_where_it_leaves_it(c, atol, t_end):
    """P' = -c from 0 leaves its band at exactly t = atol/c. Read at 0 inside
    it, P was stepped on at ever shorter steps past that, each cut back into
    the band, and only the timeout ended the run (still running at 150 s with
    none set). It is refused there, by name, as a run without the band fails
    at once. The step it is refused on starts at the band's edge, so the time
    named is the crossing: refused from further inside, the t_end = 20 case
    went at t = 3.8, on a step tried out to t = 38."""
    with pytest.raises(bngsim.SimulationError, match="carried concentration 'P' from") as info:
        _run(
            _sinking(c, "AP"),
            {"t_span": (0, t_end), "n_points": 11, "rtol": 1e-8, "atol": atol, "timeout": 20},
        )
    named = re.search(r"\(its atol [^)]*\) by t=([0-9.eE+-]*[0-9])", str(info.value))
    assert named is not None
    # The edge is a relative 2^-20 of atol, so the step starts within about
    # 1e-6 of t = atol/c; 1e-4 allows for the step's own length.
    assert float(named.group(1)) == pytest.approx(atol / float(c), rel=1e-4)


def test_each_slot_is_read_against_its_own_atol():
    """P is held at -5e-10: inside an atol of 1e-6, past one of 1e-12. Its
    neighbour A's atol is the other number, so a band read from the wrong
    slot reads P the wrong way in both runs."""
    model = _held("-5e-10")
    assert list(model.species_names) == ["A", "P"]
    x = _run(model, {**_RUN, "atol": [1e-12, 1e-6]})
    np.testing.assert_array_equal(x["A"], 1.0)
    np.testing.assert_array_equal(x["P"], -5e-10)
    with pytest.raises(bngsim.SimulationError):
        _run(_held("-5e-10"), {**_RUN, "atol": [1e-6, 1e-12]})


@pytest.mark.parametrize(("p", "inside"), [("-5e-10", True), ("-1e-9", False), ("-3e-9", False)])
def test_the_band_is_open_and_one_atol_wide(p, inside):
    """Against an atol of 1e-9: half an atol below 0 is inside the band, and
    one atol below it, or three, is not."""
    if inside:
        np.testing.assert_array_equal(_run(_held(p), _EXACT)["A"], 1.0)
    else:
        with pytest.raises(bngsim.SimulationError):
            _run(_held(p), _EXACT)


@pytest.mark.parametrize(
    "decl",
    ["species A = 1, D = -5e-10;", "species A = 1;\n  species $D = -5e-10;"],
    ids=["declared negative", "fixed"],
)
def test_a_slot_declared_negative_or_fixed_inside_its_atol_is_read_as_zero(decl):
    """The mask leaves alone more than rate-rule targets (#706): a species
    declared below 0, and a fixed one. Neither is moved by the run."""
    model = bngsim.Model.from_antimony_string(
        f"model m\n  {decl}\n  k = 0.1;\n  J: A -> ; k*A*D^0.5;\nend"
    )
    x = _run(model, _EXACT)
    np.testing.assert_array_equal(x["A"], 1.0)
    np.testing.assert_array_equal(x["D"], -5e-10)


@pytest.mark.parametrize("order", ["AP", "PA"])
@pytest.mark.parametrize("jacobian", ["auto", "fd"])
def test_the_steady_state_run_test_reads_the_band_as_the_time_course_does(order, jacobian):
    """The run test (#1357) retries a NaN f at the clamped state as the time
    course does, and now reads the band too. Without it, the state the time
    course runs from to 1e6 with A = 1 was refused: residual 5e-31, the run
    from it measured inf."""
    ss = bngsim.Simulator(_sinking("1e-30", order), method="ode", jacobian=jacobian).steady_state(
        method="integration"
    )
    assert ss.converged
    assert ss.continued_run_distance <= 1.0


def test_model1812040005_runs_in_either_order(data_dir: Path, tmp_path: Path):
    libsbml = pytest.importorskip("libsbml")
    doc = libsbml.readSBMLFromFile(str(data_dir / "MODEL1812040005.xml"))
    species = doc.getModel().getListOfSpecies()
    items = [species.remove(0) for _ in range(species.size())]
    for s in reversed(items):
        species.appendAndOwn(s)
    reversed_path = tmp_path / "reversed.xml"
    libsbml.writeSBMLToFile(doc, str(reversed_path))
    run = {"t_span": (0, 1), "n_points": 101, "rtol": 1e-6, "atol": 1e-9}
    out = []
    for path in (data_dir / "MODEL1812040005.xml", reversed_path):
        r = bngsim.Simulator(bngsim.Model.from_sbml(str(path)), method="ode").run(**run)
        names = list(r.species_names)
        out.append({n: np.asarray(r.species)[:, names.index(n)] for n in names})
    # Each species within 1e-8 of its own peak, and P (whose peak is 0) within
    # the run's atol: what the two orders leave apart is rounding.
    for name, x in out[0].items():
        bound = 1e-8 * float(np.abs(x).max()) + run["atol"]
        assert np.abs(x - out[1][name]).max() <= bound, name
