"""Work that depends only on a model's structure is done once for a model and
its clones (issue #979).

A fitting loop clones one base model for every evaluation and builds a
``Simulator`` on the clone. Two things were then done again each time, both
with sympy and both functions of the model's text alone:

- the analysis behind the output sensitivities of the global functions
  (``_codegen._analyze_output_sens``, issue #198), which a sensitivity run
  attaches to its result as the map a selector's error message is read from.
  It was memoized in a slot of each model that a clone took a copy of, so a
  clone of a model that had never been analyzed had nothing, and neither had
  the next one: 2.2 s a run for 300 functions, for a run of milliseconds;
- the guard of a rate law with a logarithm (``_jacobian.guard_rate_law_text``,
  issue #333), which every ``Model`` runs when it is made: a law that has a
  logarithm and needs no guard was parsed for every clone.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

pytest.importorskip("sympy")

# Two functions with a logarithm that need no guard, one that does, and a
# derived parameter that one of them reads. (The constants are this file's own:
# the guard is memoized by the text, for the whole process.)
NET = """begin parameters
    1 k   1.0
    2 a   0.5
    3 b   0.25
    4 K   0.1979
    5 d   a+b
end parameters
begin functions
    1 f0() ln((1+K*exp(2.0979-(a+b)))/(1+K*exp(-(a+b))))
    2 f1() ln(1.0979+K*exp(d))
    3 rate() k+0*f0()+0*f1()
    4 g() 1.0979*k*Xobs^a*ln(Xobs)
end functions
begin species
    1 X() 0
end species
begin reactions
    1 0 1 rate
end reactions
begin groups
    1 Xobs 1
end groups
"""


def _model(tmp_path, text=NET, name="model.net"):
    path = tmp_path / name
    path.write_text(text)
    return bngsim.Model.from_net(str(path))


def _analyses_made(monkeypatch) -> list:
    """Every model the analysis is computed for from here on, in order."""
    from bngsim import _codegen

    real = _codegen._compute_output_sens_analysis
    made: list = []

    def counted(model, core):
        made.append(model)
        return real(model, core)

    monkeypatch.setattr(_codegen, "_compute_output_sens_analysis", counted)
    return made


def test_clones_of_a_model_that_was_never_analyzed_share_one_analysis(tmp_path, monkeypatch):
    """The loop of the issue: the base model is cloned and never run itself.
    The analysis made for the first clone serves the second, and the base. It
    was made three times."""
    from bngsim._codegen import output_sens_support

    made = _analyses_made(monkeypatch)
    base = _model(tmp_path)
    first = output_sens_support(base.clone())
    second = output_sens_support(base.clone())
    assert len(made) == 1
    assert output_sens_support(base) == first == second
    assert len(made) == 1
    assert set(first) == {"f0", "f1", "rate", "g"}


def test_a_sensitivity_run_on_each_clone_analyzes_once(tmp_path, monkeypatch):
    """The same through ``Simulator.run``, which attaches the support map to
    every result of a sensitivity run: two clones, two Simulators, two runs
    with the same columns, one analysis."""
    made = _analyses_made(monkeypatch)
    base = _model(tmp_path)
    columns = []
    for _ in range(2):
        sim = bngsim.Simulator(base.clone(), method="ode", sensitivity_params=["k", "a"])
        result = sim.run(t_span=(0.0, 2.0), n_points=3)
        assert result._expression_sens_support.keys() == {"f0", "f1", "rate", "g"}
        columns.append(np.asarray(result.sensitivities))
    assert len(made) == 1
    np.testing.assert_array_equal(columns[0], columns[1])
    np.testing.assert_allclose(columns[0][-1, 0], [2.0, 0.0], atol=1e-9)


def test_the_support_map_of_a_clone_is_that_of_the_model_loaded_again(tmp_path):
    """Control. What a clone is served is what a model loaded from the same
    text computes for itself."""
    from bngsim._codegen import output_sens_support

    base = _model(tmp_path)
    output_sens_support(base.clone())
    served = output_sens_support(base.clone())
    again = output_sens_support(_model(tmp_path, name="again.net"))
    assert served == again


def test_a_clone_with_a_derived_parameter_pinned_has_its_own_analysis(tmp_path, monkeypatch):
    """The chain rule through a derived parameter is in the analysis while the
    parameter is attached and not once a ``set_param`` has pinned it (issue
    #188), so a pinned clone does not take the family's. Both are kept: a
    batch with a pinned row and an attached one does not analyze again for
    each."""
    from bngsim._codegen import _analyze_output_sens

    made = _analyses_made(monkeypatch)
    base = _model(tmp_path)
    pinned = base.clone()
    pinned.set_param("d", 0.9)
    attached = _analyze_output_sens(base)
    detached = _analyze_output_sens(pinned)
    assert attached is not detached and len(made) == 2
    assert set(attached["derived_expansion"]) != set(detached["derived_expansion"])
    for _ in range(2):
        assert _analyze_output_sens(base.clone()) is attached
        again = base.clone()
        again.set_param("d", 0.7)
        assert _analyze_output_sens(again) is detached
    assert len(made) == 2


def test_a_table_function_added_to_one_clone_is_in_its_key(tmp_path):
    """A table function can be added after load, and a function that reads
    one is deferred. The key had the number of functions and no table in it,
    so a model given a table function kept the analysis it had."""
    from bngsim._codegen import _output_sens_analysis_key

    base = _model(tmp_path)
    one, other = base.clone(), base.clone()
    one.add_table_function("drive", times=[0.0, 1.0], values=[0.0, 1.0])
    other.add_table_function("dose", times=[0.0, 1.0], values=[0.0, 1.0])
    keys = {_output_sens_analysis_key(m._core) for m in (base, one, other)}
    assert len(keys) == 3


def test_a_family_keeps_a_few_analyses_and_no_more(tmp_path, monkeypatch):
    """One for each key, the oldest dropped past the limit."""
    from bngsim import _codegen

    base = _model(tmp_path)
    count = iter(range(100))
    monkeypatch.setattr(_codegen, "_output_sens_analysis_key", lambda core: ("key", next(count)))
    monkeypatch.setattr(_codegen, "_compute_output_sens_analysis", lambda model, core: {})
    for _ in range(3 * _codegen._OUTPUT_SENS_ANALYSES_KEPT):
        _codegen._analyze_output_sens(base.clone())
    limit = _codegen._OUTPUT_SENS_ANALYSES_KEPT
    assert sorted(k[1] for k in base._output_sens_analysis) == list(range(2 * limit, 3 * limit))


def _parses(monkeypatch) -> list:
    from bngsim import _jacobian

    real = _jacobian._exprtk_to_sympy
    parsed: list = []

    def counted(text, *args, **kwargs):
        parsed.append(text)
        return real(text, *args, **kwargs)

    monkeypatch.setattr(_jacobian, "_exprtk_to_sympy", counted)
    return parsed


def test_a_rate_law_is_parsed_for_its_guard_once(tmp_path, monkeypatch):
    """Three functions here have a logarithm and no conditional, and are
    parsed to see whether they need the guard. Not again for a clone, whose
    functions are the same texts, or for the same model loaded again. Each
    was parsed for every one of the five: 15 parses."""
    text = NET.replace("0979", "0980")  # texts no other test has had parsed
    parsed = _parses(monkeypatch)
    base = _model(tmp_path, text)
    assert len(parsed) == 3
    for _ in range(3):
        base.clone()
    _model(tmp_path, text, name="again.net")
    assert len(parsed) == 3


def test_a_guard_read_from_the_memo_is_still_applied(tmp_path):
    """Control. ``g`` is ``1.1·k·X^a·ln(X)``, which is rewritten to its limit at
    X = 0. A model whose laws were all parsed before, and a clone, have the
    same rewrite as the first model that had them parsed."""
    text = NET.replace("0979", "0981")
    first = _model(tmp_path, text)
    second = _model(tmp_path, text, name="again.net")
    assert [name for name, _, _ in first._guarded_functions] == ["g"]
    assert "if(" in first._guarded_functions[0][2]
    assert second._guarded_functions == first._guarded_functions
    assert first.clone()._guarded_functions in ([], first._guarded_functions)
    for model in (first, second, first.clone()):
        result = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 1.0), n_points=3)
        assert np.all(np.isfinite(np.asarray(result.species)))
