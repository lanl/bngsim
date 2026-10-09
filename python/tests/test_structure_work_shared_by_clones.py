"""Work that depends only on a model's structure is done once for a model and
its clones (issue #979).

A fitting loop clones one base model for every evaluation and builds a
``Simulator`` on the clone. Two things were then done again each time, both
with sympy, and both with the answer the last clone had:

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
# derived parameter that one of them reads.
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


def test_a_models_own_analysis_outlasts_the_familys_memo(tmp_path, monkeypatch):
    """Control. A model's analysis is its own until its key changes, whatever
    else is analyzed in the family: the artifact it was built with holds what that
    analysis made of each function, a cut by the derivation budget included,
    and the support map read for its results has to be that analysis and not
    a second one. With the memo shared and nothing else, enough siblings under
    other keys dropped it, and the model analyzed again (a model cut by its
    budget then returned NaN for a function its support map called
    supported)."""
    from bngsim import _codegen

    made = _analyses_made(monkeypatch)
    kept = getattr(_codegen, "_OUTPUT_SENS_ANALYSES_KEPT", 8)
    base = _model(tmp_path)
    mine = base.clone()
    analysis = _codegen._analyze_output_sens(mine)
    for i in range(kept + 2):
        sibling = base.clone()
        sibling.set_param("a", 0.5 + i)  # a value: not in the key
        sibling.add_table_function(f"table_{i}", times=[0.0, 1.0], values=[0.0, 1.0])
        _codegen._analyze_output_sens(sibling)
    family = getattr(mine, "_output_sens_family", {})
    assert _codegen._output_sens_analysis_key(mine._core) not in family
    before = len(made)
    assert _codegen._analyze_output_sens(mine) is analysis
    # And so has a clone of it, as a clone of an analyzed model always had.
    assert _codegen._analyze_output_sens(mine.clone()) is analysis
    assert len(made) == before


# `drive` is a function, and `g` reads it. (Two observables, for a table to be
# read over either.)
READS_A_FUNCTION = """begin parameters
    1 k   1.0
    2 a   0.5
end parameters
begin functions
    1 drive() k
    2 g() drive()*a
    3 rate() k
end functions
begin species
    1 X() 0
end species
begin reactions
    1 0 1 rate
end reactions
begin groups
    1 Xobs 1
    2 Yobs 1
end groups
"""


def _output_sensitivity(model, name):
    model.reset()
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
    result = sim.run(t_span=(0.0, 2.0), n_points=3)
    return np.asarray(result.output_sensitivities([name])).reshape(3, -1)[:, 0]


def test_a_table_function_added_over_an_analyzed_function_is_seen(tmp_path):
    """A model that was analyzed and is then given a table function with the
    name of one of its functions. The compiled code reads ``drive`` from the
    table from then on, and it has no derivative in ``k``. The analysis was
    kept, its key having the number of functions in it and no table, and
    ``d drive/dk = 1`` came back with ``dg/dk = 0.5``. The selector raises
    with the reason."""
    model = _model(tmp_path, READS_A_FUNCTION)
    np.testing.assert_allclose(_output_sensitivity(model, "drive"), 1.0)
    np.testing.assert_allclose(_output_sensitivity(model, "g"), 0.5)
    model.add_table_function("drive", times=[0.0, 10.0], values=[0.0, 10.0])
    with pytest.raises(ValueError, match=r"expression 'drive' has no output sensitivity"):
        _output_sensitivity(model, "drive")


def test_a_table_function_on_one_clone_is_not_its_siblings(tmp_path):
    """Two clones of one model, one given the table over ``drive``: each has
    what its own functions are. The memo being the family's, the key has to
    tell them apart, by each table's name, by what it is read over (the same
    name over the time and over a parameter are two calls in the code), and
    by their order, which is what the code calls them by."""
    from bngsim._codegen import _output_sens_analysis_key

    base = _model(tmp_path, READS_A_FUNCTION)
    plain, tabled = base.clone(), base.clone()
    tabled.add_table_function("drive", times=[0.0, 10.0], values=[0.0, 10.0])
    with pytest.raises(ValueError, match=r"expression 'drive' has no output sensitivity"):
        _output_sensitivity(tabled, "drive")
    np.testing.assert_allclose(_output_sensitivity(plain, "drive"), 1.0)
    np.testing.assert_allclose(_output_sensitivity(base.clone(), "g"), 0.5)

    def keyed(*tables):
        model = base.clone()
        for name, index in tables:
            model.add_table_function(name, times=[0.0, 1.0], values=[0.0, 1.0], index=index)
        return _output_sens_analysis_key(model._core)

    keys = [
        keyed(),
        keyed(("one", "time")),
        keyed(("one", "a")),
        keyed(("one", "k")),
        keyed(("one", "Xobs")),
        keyed(("one", "Yobs")),
        keyed(("one", "time"), ("two", "time")),
        keyed(("two", "time"), ("one", "time")),
    ]
    assert len(set(keys)) == len(keys)


def test_what_the_tables_are_read_over_is_asked_once_for_each_list_of_them(tmp_path):
    """The key has what each table function is read over, which is in
    ``codegen_data()``: the whole model, 0.1 s for one of 58,000 reactions,
    and the key is asked for on every sensitivity run. It is kept on the
    model for the list of table names it was read for, and a clone has it."""
    from bngsim._codegen import _output_sens_analysis_key, _table_function_bindings

    model = _model(tmp_path, READS_A_FUNCTION)
    assert _table_function_bindings(model._core, model) == ()
    assert model._table_function_bindings is None
    model.add_table_function("one", times=[0.0, 1.0], values=[0.0, 1.0], index="k")
    first = _table_function_bindings(model._core, model)
    assert [binding[:2] for binding in first] == [("one", "parameter")]
    assert _table_function_bindings(model._core, model) is first
    clone = model.clone()
    assert _table_function_bindings(clone._core, clone) is first
    clone.add_table_function("two", times=[0.0, 1.0], values=[0.0, 1.0])
    second = _table_function_bindings(clone._core, clone)
    assert [binding[0] for binding in second] == ["one", "two"]
    assert _table_function_bindings(model._core, model) is first
    # Asked with no model to keep it on, it is the same answer.
    assert _table_function_bindings(clone._core) == second
    assert _output_sens_analysis_key(clone._core) == _output_sens_analysis_key(clone._core, clone)
    # And the analysis asks with its model: a model that was only analyzed has it kept.
    from bngsim._codegen import output_sens_support

    other = _model(tmp_path, READS_A_FUNCTION, name="other.net")
    other.add_table_function("one", times=[0.0, 1.0], values=[0.0, 1.0], index="k")
    output_sens_support(other)
    assert other._table_function_bindings == (("one",), first)


def test_models_that_are_not_clones_share_nothing(tmp_path, monkeypatch):
    """Control. Two models of one shape, loaded apart, with different
    functions: the memo is a family's, and each is analyzed for itself."""
    from bngsim import _codegen

    made = _analyses_made(monkeypatch)
    one = _model(tmp_path, READS_A_FUNCTION, name="one.net")
    other = _model(
        tmp_path, READS_A_FUNCTION.replace("drive()*a", "drive()*a*a"), name="other.net"
    )
    assert _codegen._output_sens_analysis_key(one._core) == _codegen._output_sens_analysis_key(
        other._core
    )
    assert getattr(one, "_output_sens_family", 1) is not getattr(other, "_output_sens_family", 2)
    np.testing.assert_allclose(_output_sensitivity(one, "g"), 0.5)
    np.testing.assert_allclose(_output_sensitivity(other, "g"), 0.25)
    assert len(made) == 2


def test_a_family_keeps_a_few_analyses_and_no_more(tmp_path, monkeypatch):
    """One for each key, the oldest dropped past the limit."""
    from bngsim import _codegen

    base = _model(tmp_path)
    count = iter(range(100))
    monkeypatch.setattr(
        _codegen, "_output_sens_analysis_key", lambda core, model=None: ("key", next(count))
    )
    monkeypatch.setattr(_codegen, "_compute_output_sens_analysis", lambda model, core: {})
    for _ in range(3 * _codegen._OUTPUT_SENS_ANALYSES_KEPT):
        _codegen._analyze_output_sens(base.clone())
    limit = _codegen._OUTPUT_SENS_ANALYSES_KEPT
    assert sorted(k[1] for k in base._output_sens_family) == list(range(2 * limit, 3 * limit))


def test_an_analysis_made_meanwhile_is_the_one_that_is_taken(tmp_path, monkeypatch):
    """Two clones analyzed at once under one key each derive, and the one
    that finishes second takes the first one's analysis, for itself and for
    the family: one answer under one key."""
    from bngsim import _codegen

    base = _model(tmp_path)
    first, second = base.clone(), base.clone()
    theirs = {"made by": "the other thread"}

    def while_the_other_finishes(model, core):
        model._output_sens_family[_codegen._output_sens_analysis_key(core)] = theirs
        return {"made by": "this one"}

    monkeypatch.setattr(_codegen, "_compute_output_sens_analysis", while_the_other_finishes)
    assert _codegen._analyze_output_sens(first) is theirs
    assert _codegen._analyze_output_sens(second) is theirs
    assert list(base._output_sens_family.values()) == [theirs]


def test_clones_analyzed_at_once_leave_the_memo_whole(tmp_path, monkeypatch):
    """Clones in several threads, each asking for the analysis under a key
    they share and under one of its own, with the family's memo full: the
    memo is changed under a lock, nobody's asking fails, and it stays within
    its limit. (With no lock, a thread dropping the oldest while another did
    raised ``KeyError`` and ``dictionary changed size during iteration``, 19
    times in 4,186 analyses at a switch interval of a microsecond.)"""
    import sys
    import threading

    from bngsim import _codegen

    base = _model(tmp_path)
    monkeypatch.setattr(_codegen, "_OUTPUT_SENS_ANALYSES_KEPT", 2)
    monkeypatch.setattr(_codegen, "_compute_output_sens_analysis", lambda model, core: {"of": 1})
    shared_key = ("shared",)
    local = threading.local()
    monkeypatch.setattr(
        _codegen,
        "_output_sens_analysis_key",
        lambda core, model=None: getattr(local, "key", shared_key),
    )
    errors, given = [], []

    def work(n):
        try:
            for i in range(300):
                local.key = shared_key
                given.append(_codegen._analyze_output_sens(base.clone()))
                local.key = ("own", n, i)
                _codegen._analyze_output_sens(base.clone())
        except Exception as error:  # noqa: BLE001 - what the test is for
            errors.append(repr(error))

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
    finally:
        sys.setswitchinterval(interval)
    assert errors == []
    assert len(given) == 8 * 300 and len(base._output_sens_family) <= 2


def _parses(monkeypatch) -> list:
    from bngsim import _jacobian

    real = _jacobian._exprtk_to_sympy
    parsed: list = []

    def counted(text, *args, **kwargs):
        parsed.append(text)
        return real(text, *args, **kwargs)

    monkeypatch.setattr(_jacobian, "_exprtk_to_sympy", counted)
    return parsed


@pytest.fixture
def fresh_guard_memo(monkeypatch):
    """The guard's memo is the process's: each test here starts it empty. (On
    a build without one this sets a name nothing reads.)"""
    from bngsim import _jacobian

    memo: dict = {}
    monkeypatch.setattr(_jacobian, "_GUARD_MEMO", memo, raising=False)
    return memo


def test_a_rate_law_is_parsed_for_its_guard_once(tmp_path, monkeypatch, fresh_guard_memo):
    """Three functions here have a logarithm and no conditional, and are
    parsed to see whether they need the guard. Not again for a clone, whose
    functions are the same texts, or for the same model loaded again. Each
    was parsed for every one of the five: 15 parses."""
    parsed = _parses(monkeypatch)
    base = _model(tmp_path)
    assert len(parsed) == 3
    for _ in range(3):
        base.clone()
    _model(tmp_path, name="again.net")
    assert len(parsed) == 3


def test_a_guard_read_from_the_memo_is_still_applied(tmp_path, fresh_guard_memo):
    """Control. ``g`` is ``1.1·k·X^a·ln(X)``, which is rewritten to its limit
    at X = 0, and here the rate reads it: unguarded, the rate is no number at
    the start, where X is 0, and neither is anything after. A model whose laws
    were all parsed before, and a clone, have the same rewrite as the first
    model that had them parsed, and run."""
    text = NET.replace("    3 rate() k+0*f0()+0*f1()\n", "    3 rate() k+0*f0()+0*f1()+g()\n")
    assert "+g()" in text
    first = _model(tmp_path, text)
    second = _model(tmp_path, text, name="again.net")
    assert [name for name, _, _ in first._guarded_functions] == ["g"]
    assert "if(" in first._guarded_functions[0][2]
    assert second._guarded_functions == first._guarded_functions
    for model in (first, second, first.clone()):
        result = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 1.0), n_points=3)
        species = np.asarray(result.species)
        assert np.all(np.isfinite(species)) and species[-1, 0] > 0.1


@pytest.mark.parametrize("step", ["_exprtk_to_sympy", "sympy_to_exprtk"])
def test_a_text_that_could_not_be_decided_is_asked_again(monkeypatch, fresh_guard_memo, step):
    """The parser and the writer each return nothing on any exception, and
    running out of stack is one (it is the writer that does, in a deep call
    stack): a text first met there was not guarded, and with that memoized it
    would not be for the rest of the process. Only an answer that was reached
    is kept."""
    from bngsim import _jacobian

    text = "vmax*Atot^n*ln(Atot)"
    real = getattr(_jacobian, step)
    monkeypatch.setattr(_jacobian, step, lambda expr: None)
    assert _jacobian.guard_rate_law_text(text) is None
    assert text not in fresh_guard_memo
    monkeypatch.setattr(_jacobian, step, real)
    guarded = _jacobian.guard_rate_law_text(text)
    assert guarded is not None and "if(" in guarded
    assert fresh_guard_memo[text] == guarded


def test_a_clone_has_its_parents_guard_and_does_not_ask(tmp_path, monkeypatch, fresh_guard_memo):
    """A clone's core has the parent's functions as the guard left them, and
    the clone takes the parent's list of them: it does not decide again. A
    parent whose guard could not be decided (loaded deep in a call stack) and
    a clone made where it could be were two models of one family with
    different functions, and one memo of their analysis between them."""
    from bngsim import _jacobian

    def functions(model):
        return [f.get("eval_expression") for f in model._core.codegen_data()["functions"]]

    real = _jacobian._exprtk_to_sympy
    monkeypatch.setattr(_jacobian, "_exprtk_to_sympy", lambda expr: None)
    undecided = _model(tmp_path)
    assert undecided._guarded_functions == []
    monkeypatch.setattr(_jacobian, "_exprtk_to_sympy", real)
    parsed = _parses(monkeypatch)
    clone = undecided.clone()
    assert clone._guarded_functions == [] and functions(clone) == functions(undecided)
    decided = _model(tmp_path, name="again.net")
    assert [name for name, _, _ in decided._guarded_functions] == ["g"]
    assert len(parsed) == 3
    twin = decided.clone()
    assert twin._guarded_functions == decided._guarded_functions
    assert twin._guarded_functions is not decided._guarded_functions
    assert functions(twin) == functions(decided) != functions(undecided)
    assert len(parsed) == 3


def test_the_guards_memo_is_emptied_when_it_is_full(monkeypatch, fresh_guard_memo):
    """It holds a text for every law with a logarithm that the process has
    met, and no more than its limit of them."""
    from bngsim import _jacobian

    monkeypatch.setattr(_jacobian, "_GUARD_MEMO_MAX", 3)
    for i in range(10):
        assert _jacobian.guard_rate_law_text(f"k{i}*ln(K{i})") is None
        assert len(fresh_guard_memo) <= 3
    assert "k9*ln(K9)" in fresh_guard_memo
