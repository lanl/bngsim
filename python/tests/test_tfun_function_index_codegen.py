"""A table function indexed by a function is read at the function's value under codegen (#965).

``g() tfun([0,100],[0,100], h)`` with ``h() 2*Atot``: the table spec named the
index by the slot the function ``h`` owns, ``index_kind == "parameter"``, and the
compiled RHS read that slot, ``p[idx]``. Only the interpreter's
``evaluate_functions`` writes it, so compiled code saw ``h``'s load-time value and
the trajectory froze at its initial state, with nothing raised. That hit
``codegen=True`` and the default path at or above ``BNGSIM_CODEGEN_THRESHOLD``.
The core now reports such a table as ``index_kind == "function"``, and the
compiled RHS reads ``func[h]``, computed before the table call.

The table is the identity, so g = h = 2A, dA/dt = -2A^2 and A(t) = 10/(1 + 20t).
"""

from __future__ import annotations

import bngsim
import bngsim._codegen as cg
import numpy as np
import pytest

NET = """begin parameters
end parameters
begin species
    1 A() 10
    2 B() 0
end species
begin reactions
    1 1 2 g
end reactions
begin groups
    1 Atot 1
end groups
begin functions
{functions}
end functions
"""

SPELLINGS = {
    "whole body": "    1 g() tfun([0,100],[0,100], h)\n    2 h() 2*Atot",
    "wrapped in arithmetic": "    1 g() tfun([0,100],[0,100], h) + 0\n    2 h() 2*Atot",
    "index declared first": "    1 h() 2*Atot\n    2 g() tfun([0,100],[0,100], h)",
}

TIMES = np.linspace(0.0, 1.0, 5)
CLOSED_FORM = 10.0 / (1.0 + 20.0 * TIMES)


@pytest.fixture(autouse=True)
def _cold_codegen_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")


def _model(tmp_path, functions: str) -> bngsim.Model:
    net = tmp_path / "tfun_function_index.net"
    net.write_text(NET.format(functions=functions))
    return bngsim.Model.from_net(str(net))


def _a(sim: bngsim.Simulator) -> np.ndarray:
    result = sim.run(t_span=(0.0, 1.0), n_points=len(TIMES), rtol=1e-10, atol=1e-12)
    return np.asarray(result.species)[:, 0]


@pytest.mark.parametrize("functions", SPELLINGS.values(), ids=SPELLINGS.keys())
def test_codegen_true_follows_the_closed_form(tmp_path, functions):
    sim = bngsim.Simulator(_model(tmp_path, functions), method="ode", codegen=True)
    assert sim.codegen_backend in ("cc", "mir")
    np.testing.assert_allclose(_a(sim), CLOSED_FORM, rtol=1e-7)


@pytest.mark.parametrize("functions", SPELLINGS.values(), ids=SPELLINGS.keys())
def test_default_path_above_the_threshold_follows_the_closed_form(
    tmp_path, monkeypatch, functions
):
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
    sim = bngsim.Simulator(_model(tmp_path, functions), method="ode")
    assert sim.codegen_backend in ("cc", "mir")
    np.testing.assert_allclose(_a(sim), CLOSED_FORM, rtol=1e-7)


@pytest.mark.parametrize("functions", SPELLINGS.values(), ids=SPELLINGS.keys())
def test_compiled_and_interpreted_agree(tmp_path, functions):
    # A model per run: a run leaves its end state in the model it was given.
    compiled = _a(bngsim.Simulator(_model(tmp_path, functions), method="ode", codegen=True))
    interpreted = _a(bngsim.Simulator(_model(tmp_path, functions), method="ode", codegen=False))
    np.testing.assert_allclose(compiled, interpreted, rtol=1e-7)


def test_the_core_reports_a_function_index_as_its_own_kind(tmp_path):
    # Reported as the slot h's value is written to, every reader had to tell
    # for itself that a function owns it, and one that did not read the
    # load-time value. Both listings, and the analytic Jacobian, which must
    # not take the table for a constant: it moves with h, and h with A.
    core = _model(tmp_path, SPELLINGS["whole body"])._core
    h = list(core.function_names).index("h")
    expected = {"index_kind": "function", "index_func_idx": h}
    expected |= {"index_param_idx": -1, "index_obs_idx": -1}
    for (spec,) in (core.codegen_data()["table_functions"], core.table_function_tables):
        assert {key: spec[key] for key in expected} == expected
    assert "tfun_g" not in core.functional_jacobian_context()["constant_names"]


def test_the_index_reads_the_functions_value_and_orders_it_first(tmp_path):
    data = _model(tmp_path, SPELLINGS["whole body"])._core.codegen_data()
    calls = cg._tfun_call_map(data)
    h = [f["name"] for f in data["functions"]].index("h")
    assert calls == {"g": (0, f"func[{h}]")}
    order = cg._topological_function_order(data["functions"], calls)
    assert order.index(h) < order.index(1 - h)


def test_a_table_over_a_plain_parameter_still_reads_its_slot(tmp_path):
    net = NET.replace(
        "begin parameters\nend parameters", "begin parameters\n    1 x 5\nend parameters"
    ).format(functions="    1 g() tfun([0,100],[0,100], x)")
    path = tmp_path / "tfun_parameter_index.net"
    path.write_text(net)
    core = bngsim.Model.from_net(str(path))._core
    data = core.codegen_data()
    x = [p["name"] for p in data["parameters"]].index("x")
    assert cg._tfun_call_map(data) == {"g": (0, f"p[{x}]")}
    (spec,) = data["table_functions"]
    assert (spec["index_kind"], spec["index_param_idx"], spec["index_func_idx"]) == (
        "parameter",
        x,
        -1,
    )
    assert "tfun_g" in core.functional_jacobian_context()["constant_names"]


def test_the_recorded_function_values_read_the_index_live(tmp_path):
    # The compiled output evaluator records g and h at each row: both are 2A.
    sim = bngsim.Simulator(_model(tmp_path, SPELLINGS["whole body"]), method="ode", codegen=True)
    result = sim.run(t_span=(0.0, 1.0), n_points=len(TIMES), rtol=1e-10, atol=1e-12)
    for name in ("g", "h"):
        np.testing.assert_allclose(result.expressions[name], 2.0 * CLOSED_FORM, rtol=1e-7)


@pytest.mark.parametrize(
    "functions",
    [
        "    1 g() tfun([0,100],[0,100], g)",
        "    1 g() tfun([0,100],[0,100], h)\n    2 h() 2*Atot + 0*g()",
    ],
    ids=["indexed by itself", "index reads the table"],
)
def test_a_table_its_index_reads_declines_codegen(tmp_path, monkeypatch, functions):
    # No order computes g's index before g: kept as a self-edge, the compiled C
    # read func[g] uninitialised and returned whatever the stack held.
    data = _model(tmp_path, functions)._core.codegen_data()
    with pytest.raises(
        cg.CodegenDeclined,
        match=r"reference each other.*a cyclic function graph has no assignment order",
    ):
        cg._topological_function_order(data["functions"], cg._tfun_call_map(data))
    with pytest.raises(RuntimeError, match="codegen declined"):
        bngsim.Simulator(_model(tmp_path, functions), method="ode", codegen=True)
    # Above the threshold the default path declines to the interpreter.
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
    default = bngsim.Simulator(_model(tmp_path, functions), method="ode")
    assert default.codegen_backend not in ("cc", "mir")


def test_a_table_whose_name_ends_in_another_tables_call_takes_no_edge_from_it():
    # ``tfun_a_tfun_g()`` holds ``tfun_g(`` as a substring; g's index is h, and h
    # reads a_tfun_g, so a substring match ordered a_tfun_g after h: a cycle.
    functions = [
        {"name": "a_tfun_g", "expression": "tfun_a_tfun_g()"},
        {"name": "h", "expression": "a_tfun_g()+1"},
        {"name": "g", "expression": "tfun_g()"},
    ]
    calls = {"a_tfun_g": (0, "t"), "g": (1, "func[1]")}
    assert cg._topological_function_order(functions, calls) == [0, 1, 2]


# ── Forward sensitivities: the table has no analytic sensitivity RHS ─────────
# A run takes its columns from CVODES' difference quotient, over the compiled
# right-hand side, so it reads the index live as the run does.


def test_a_smooth_index_gives_the_closed_form_sensitivity(tmp_path):
    # h() = c*Atot: A = 10/(1 + 10ct), so dA/dc = -100t/(1 + 10ct)^2.
    net = NET.replace(
        "begin parameters\nend parameters", "begin parameters\n    1 c 2\nend parameters"
    ).format(functions="    1 g() tfun([0,100],[0,100], h)\n    2 h() c*Atot")
    path = tmp_path / "tfun_smooth_index.net"
    path.write_text(net)
    run = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["c"]
    ).run(t_span=(0.0, 1.0), n_points=3, rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities)[-1, 0, 0]
    assert got == pytest.approx(-100.0 / (1.0 + 20.0) ** 2, rel=1e-6)


CROSSING_NET = """begin parameters
    1 A0 10
    2 a 0.5
    3 thr 2
end parameters
begin species
    1 A() A0
    2 Y() 0
end species
begin reactions
    1 1 0 a
    2 0 2 g
end reactions
begin groups
    1 Aobs 1
end groups
begin functions
{functions}
end functions
"""


@pytest.mark.parametrize(
    "functions",
    [
        "    1 h() if(Aobs<thr,1,0)\n    2 g() tfun([0,1],[0,3], h)",
        "    1 h0() if(Aobs<thr,1,0)\n    2 h() tfun([0,1],[0,1], h0)\n"
        "    3 g() tfun([0,1],[0,3], h)",
    ],
    ids=["index holds the condition", "index reads a table whose index holds it"],
)
def test_an_index_that_holds_a_state_crossing_is_refused(tmp_path, functions):
    # A decays at rate a past thr at t* = ln(A0/thr)/a = 3.2, where g jumps from
    # 0 to 3 and the quotient reads across it (issue #938). Run anyway, CVODE
    # made no progress at t*. Before #965 the compiled table read h as it was
    # built, never jumped, and the run returned columns that ignored the jump.
    path = tmp_path / "tfun_crossing_index.net"
    path.write_text(CROSSING_NET.format(functions=functions))
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["thr", "a"]
    )
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"Aobs<thr"):
        sim.run(t_span=(0.0, 6.0), n_points=3, rtol=1e-10, atol=1e-12)
