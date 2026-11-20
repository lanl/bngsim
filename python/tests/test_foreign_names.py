"""The calls a caller brings from another tool name bngsim's (issue #1700).

An agent that has not read the docs writes the call it knows from libRoadRunner,
Tellurium, SciPy or BioNetGen and corrects from the error, so the error has to
name the call to make. These tests pin the table those errors come from
(``bngsim._foreign_names.FOREIGN_NAMES``), check that every call it names
exists, and check the answers themselves: the two constructor crashes, the
attribute and keyword errors, ``ParameterError``'s near matches, and
``result.species["A"]``. They also pin what must not change: protocol probes
and pickling see Python's own errors, and no class grew a ``__getattr__`` or a
``__setattr__``, which would slow every successful lookup or write.
"""

from __future__ import annotations

import copy
import inspect
import pickle
import re
import traceback
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import Model, ModelError, ParameterError, Result, Simulator, _agent_guide
from bngsim import _foreign_names as fn

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
DECAY = DATA / "degenerate_decay.net"  # A() -> B() at keff = k1 + k2; observables A_tot, B_tot
FIXED = DATA / "fixed_species.net"  # $A() -> B(): A is fixed
ANTIMONY = "model m\n  J0: A -> B; kf*A\n  kf = 0.1; kr = 0.2; kcat = 1; A = 10; B = 0\nend"

CLASSES = {"Model": Model, "Simulator": Simulator, "Result": Result}


@pytest.fixture(scope="module")
def model() -> Model:
    return Model.load(DECAY)


@pytest.fixture(scope="module")
def sim(model: Model) -> Simulator:
    return Simulator(model)


@pytest.fixture(scope="module")
def result(sim: Simulator) -> Result:
    return sim.run(t_span=(0, 10), n_points=11)


@pytest.fixture
def fresh() -> Simulator:
    """A Simulator on a model no run has advanced, for a test that runs: the
    shared one's model holds the state its last run ended at (issue #1701)."""
    return Simulator(Model.load(DECAY))


def instance(owner: str, model: Model, sim: Simulator, result: Result) -> object:
    return {"Model": model, "Simulator": sim, "Result": result}[owner]


def message(exc: BaseException) -> str:
    """What a traceback prints for *exc*, "Did you mean" included."""
    return "".join(traceback.format_exception_only(exc)).strip()


# ─── The table ────────────────────────────────────────────────────────────────

#: Every name in the table, by owner. Pinned so that adding or dropping one is a
#: change a reviewer sees, and the agent guide's table (#1699), generated from
#: FOREIGN_NAMES, moves with it.
PINNED = {
    "bngsim": [
        "RoadRunner", "from_antimony", "from_antimony_string", "from_bngl", "from_net",
        "from_sbml", "from_sbml_string", "load", "loadAntimonyModel", "loadSBMLModel",
        "loada", "run", "seed", "set_seed", "simulate",
    ],
    "Model": [
        "from_antimony_file", "from_bngl_file", "from_bngl_string", "from_file",
        "from_net_file", "from_sbml_file", "from_string", "getFloatingSpeciesIds",
        "getGlobalParameterIds", "getValue", "get_parameter", "n_params", "observables",
        "parameters", "resetAll", "resetConcentrations", "run", "saveConcentrations",
        "setConcentration", "setParameter", "setValue", "set_parameter", "set_parameters",
        "simulate", "species",
    ],
    "Simulator": [
        "atol", "getFloatingSpeciesIds", "getGlobalParameterIds", "getSteadyStateValues",
        "getValue", "get_concentration", "get_param", "gillespie", "integrate", "integrator",
        "observable_names", "param_names", "reset", "resetAll", "result", "rtol", "seed",
        "setIntegrator", "setSeed", "setValue", "set_concentration", "set_param",
        "set_parameter", "set_params", "set_seed", "simulate", "solve", "species_names",
        "steadyState", "timeCourseSelections",
    ],
    "Result": [
        "as_dataframe", "colnames", "df", "message", "plot", "status", "success",
        "t", "times", "to_dataframe", "to_df", "to_numpy", "to_pandas", "y",
    ],
    "Simulator.run": [
        "duration", "end", "integrator", "method", "n_steps", "points", "selections", "steps",
        "t_end", "t_eval", "t_start", "tend", "times", "tspan", "y0",
    ],
    "Simulator.parameter_scan": [
        "sample_times", "scan_values", "start", "suffix", "t_eval", "times", "values",
    ],
    "Simulator.run_batch": ["sample_times", "start", "t_eval", "times", "y0"],
    "Simulator.run_replicates": ["sample_times", "start", "t_eval", "times"],
    "Simulator.bifurcate": ["sample_times", "t_eval", "times"],
    "Simulator.compute_all_sensitivities": ["sample_times", "t_eval", "times"],
    "Simulator.run_until": [
        "duration", "end", "max_step", "sample_times", "t_end", "t_eval", "t_start", "tend",
        "timeout", "times",
    ],
    "pattern": [
        'from bngsim import simulate', 'model.k1', 'model.k1 = value', 'model["S1"]',
        'model["k1"] = value', 'result.A', 'sim.k1 = value',
    ],
    "translation": [
        'BngSimulator(model).run(method="ssa", n_runs=n)',
        'ScipyOdeSimulator(model).run(param_values={...})',
        'ScipyOdeSimulator(model, tspan=t).run()',
        'amici.runAmiciSimulation(model, solver)',
        'basico.load_model(path)',
        'basico.run_steadystate()',
        'basico.run_time_course(duration=d, intervals=n)',
        'basico.run_time_course(method="stochastic")',
        'basico.set_parameters(name, initial_value=v)',
        'basico.set_species(name, initial_concentration=c)',
        'generate_network({overwrite=>1})',
        'model.setParameterByName(name, value)',
        'model.setTimepoints(times)',
        'parameter_scan({parameter=>"k", par_min=>a, par_max=>b, n_scan_pts=>n, log_scale=>1})',
        'rdata.x',
        'res.observables["obs"]',
        'simulate({..., continue=>1})',
        'simulate({method=>"ode", t_end=>T, n_steps=>N})',
        'solver.setSensitivityOrder(amici.SensitivityOrder.first)',
    ],
}  # fmt: skip


def test_the_table_holds_no_name_a_protocol_checks_for():
    """A descriptor is a class attribute to inspect.getattr_static, which a
    runtime_checkable Protocol reads on 3.12+: ``keys`` on Result would make
    it pass for a mapping."""
    attributes = {e.name for e in fn.FOREIGN_NAMES if e.owner in CLASSES}
    assert not attributes & fn.PROTOCOL_NAMES
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class HasKeys(Protocol):
        def keys(self) -> object: ...

    @runtime_checkable
    class Copyable(Protocol):
        def copy(self) -> object: ...

    model = Model.load(DECAY)
    result = Simulator(model).run(t_span=(0, 1), n_points=3)
    assert not isinstance(result, HasKeys)
    assert not isinstance(model, Copyable)


def test_the_table_is_pinned():
    got: dict[str, list[str]] = {}
    for entry in fn.FOREIGN_NAMES:
        got.setdefault(entry.owner, []).append(entry.name)
    assert {k: sorted(v) for k, v in got.items()} == {k: sorted(v) for k, v in PINNED.items()}
    pairs = [(e.owner, e.name) for e in fn.FOREIGN_NAMES]
    assert len(pairs) == len(set(pairs))


@pytest.mark.parametrize("entry", fn.FOREIGN_NAMES, ids=lambda e: f"{e.owner}.{e.name}")
def test_an_entry_names_something_and_is_not_a_real_name(entry):
    assert entry.use or entry.note, "an entry says what to do"
    assert entry.origin
    if entry.owner == "pattern":
        assert entry.use, "a pattern names the call to make"
        assert all(entry.name not in vars(cls) for cls in CLASSES.values())
    elif entry.owner == "translation":
        # Another tool's call, as that tool spells it: the guide's table only.
        assert entry.use, "a translation names the call to make"
        assert entry.origin in _agent_guide.TOOLS
    elif entry.owner == "bngsim":
        assert entry.name not in vars(bngsim) and entry.name not in bngsim.__all__
    elif entry.owner.startswith("Simulator."):
        method = entry.owner.removeprefix("Simulator.")
        assert method in fn.RUN_CALLS
        assert entry.name not in inspect.signature(getattr(Simulator, method)).parameters
    else:
        cls = CLASSES[entry.owner]
        # The class holds the table's descriptor and nothing else under the name.
        assert isinstance(vars(cls)[entry.name], fn._ForeignAttribute)
        assert all(entry.name not in vars(k) for k in cls.__mro__[1:])


#: ``owner.name`` references in a hint, by the object they are read on, and a
#: name read off a call's value (``sim.steady_state().concentrations``).
_REFERENCE = re.compile(
    r"\b(bngsim\.Model|bngsim\.Simulator|bngsim|Model|Simulator|sim\.model|model|sim|result)"
    r"\.([A-Za-z_]\w*)"
)
_CHAINED = re.compile(r"\.(steady_state|as_roadrunner|dataframe)(?:\(\))?\.([A-Za-z_]\w*)")
_READ_ON = {
    "bngsim": bngsim,
    "bngsim.Model": Model,
    "Model": Model,
    "model": Model,
    "sim.model": Model,
    "bngsim.Simulator": Simulator,
    "Simulator": Simulator,
    "sim": Simulator,
    "result": Result,
}


@pytest.mark.parametrize("entry", fn.FOREIGN_NAMES, ids=lambda e: f"{e.owner}.{e.name}")
def test_every_call_an_entry_names_exists(entry):
    """The hint is the table's whole point: a call it names that bngsim does
    not have would send the caller from one wrong call to another."""
    text = f"{entry.use} {entry.note}"
    for obj, attr in _REFERENCE.findall(text):
        target = _READ_ON[obj]
        if attr in ("T", "shape", "plot", "columns"):  # numpy's and pandas' own
            continue
        assert attr in dir(target) or attr in getattr(target, "__all__", ()), f"{obj}.{attr}"
        assert not isinstance(vars(target).get(attr), fn._ForeignAttribute), f"{obj}.{attr}"
    returns = {
        "steady_state": bngsim.SteadyStateResult,
        "as_roadrunner": bngsim.NamedArray(np.zeros((1, 1)), ["time"]),  # colnames is per array
        "dataframe": None,  # a pandas DataFrame
    }
    for call, attr in _CHAINED.findall(text):
        if returns[call] is not None:
            assert hasattr(returns[call], attr) or attr in dir(returns[call]), f"{call}.{attr}"
    takes = {
        "Simulator": inspect.signature(Simulator).parameters,
        **{
            name: inspect.signature(getattr(Simulator, name)).parameters
            for name in (*fn.RUN_CALLS, "steady_state", "set_tolerances")
        },
        "plot": {"x"},  # pandas
    }
    # Each call's keywords, one level of parentheses inside its arguments.
    for call, args in re.findall(r"(\w+)\(((?:[^()]|\([^()]*\))*)\)", text):
        for keyword in re.findall(r"\b([a-z_]+)=", args):
            assert keyword in takes[call], f"{call}({keyword}=)"
    if entry.owner.startswith("Simulator."):
        method = entry.owner.removeprefix("Simulator.")
        method_takes = inspect.signature(getattr(Simulator, method)).parameters
        for keyword in re.findall(r"^([a-z_]+)=", entry.use):
            assert keyword in method_takes, f"{method}({keyword}=)"


def _do_pattern(code: str, model: Model, sim: Simulator, result: Result) -> None:
    """Perform a guide-only pattern on the fixtures: k1 is a parameter of the
    decay model, S1 stands for a species name, A for an observable."""
    namespace = {"model": model, "sim": sim, "result": result, "value": 2.0}
    code = code.replace('"S1"', '"A()"').replace("result.A", "result.A_tot")
    exec(code, namespace)


@pytest.mark.parametrize("entry", fn.PATTERNS, ids=lambda e: e.name)
def test_a_guide_only_pattern_still_gets_pythons_error(entry, model, sim, result):
    """The rows bngsim lists for the guide alone fail with Python's own error,
    which names no call. Where bngsim comes to catch one with a hint, the row
    belongs under its owner instead, and this fails to say so."""
    with pytest.raises((AttributeError, TypeError, ImportError)) as err:
        _do_pattern(entry.name, model, sim, result)
    assert "Use " not in str(err.value)


# ─── Attribute errors ────────────────────────────────────────────────────────

ATTRIBUTE_ENTRIES = [e for e in fn.FOREIGN_NAMES if e.owner in CLASSES]


@pytest.mark.parametrize("entry", ATTRIBUTE_ENTRIES, ids=lambda e: f"{e.owner}.{e.name}")
def test_a_foreign_attribute_names_the_call(entry, model, sim, result):
    obj = instance(entry.owner, model, sim, result)
    cls = CLASSES[entry.owner]
    with pytest.raises(AttributeError) as read:
        getattr(obj, entry.name)
    assert str(read.value) == (
        f"'{cls.__name__}' object has no attribute '{entry.name}'. {entry.hint}"
    )
    # No "Did you mean" of Python's own beside the call the error names.
    assert "Did you mean" not in message(read.value)
    with pytest.raises(AttributeError, match=re.escape(entry.hint)):
        getattr(cls, entry.name)
    with pytest.raises(AttributeError, match=re.escape(entry.hint)):
        setattr(obj, entry.name, 1)
    assert not hasattr(obj, entry.name)
    assert getattr(obj, entry.name, None) is None
    assert entry.name not in dir(obj)
    # The class lists it, with the hint as its docstring, as help() shows it.
    assert entry.hint in vars(cls)[entry.name].__doc__


@pytest.mark.parametrize(
    "entry", [e for e in fn.FOREIGN_NAMES if e.owner == "bngsim"], ids=lambda e: e.name
)
def test_a_foreign_module_name_names_the_call(entry):
    with pytest.raises(AttributeError) as err:
        getattr(bngsim, entry.name)
    assert str(err.value) == f"module 'bngsim' has no attribute '{entry.name}'. {entry.hint}"
    assert not hasattr(bngsim, entry.name)


def test_the_issues_example_messages(model, sim, result):
    with pytest.raises(AttributeError, match=r"Use sim\.run\(t_span=\(t0, t1\), n_points=n\)"):
        sim.simulate(0, 10, 11)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match=r"Use sim\.model\.reset\(\)"):
        sim.reset()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match=r"Use result\.dataframe\. It is a property"):
        result.to_dataframe()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match=r"Use bngsim\.Model\.load\(path\)"):
        bngsim.Model.from_file(DECAY)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match=r"Use bngsim\.Model\.from_bngl\(path\)"):
        bngsim.Model.from_bngl_string("begin model\nend model")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match=r"Use bngsim\.Model\.load\(path\)"):
        bngsim.load(DECAY)  # type: ignore[attr-defined]


@pytest.mark.parametrize("owner", ["Model", "Simulator", "Result"])
def test_a_name_off_the_table_gets_pythons_own_error(owner, model, sim, result):
    obj = instance(owner, model, sim, result)
    with pytest.raises(AttributeError) as err:
        obj.not_a_bngsim_name  # type: ignore[attr-defined]  # noqa: B018
    assert str(err.value) == f"'{owner}' object has no attribute 'not_a_bngsim_name'"
    with pytest.raises(AttributeError) as err:
        type(obj).not_a_bngsim_name  # type: ignore[attr-defined]  # noqa: B018
    assert str(err.value) == f"type object '{owner}' has no attribute 'not_a_bngsim_name'"
    with pytest.raises(AttributeError) as err:
        bngsim.not_a_bngsim_name  # type: ignore[attr-defined]  # noqa: B018
    assert str(err.value) == "module 'bngsim' has no attribute 'not_a_bngsim_name'"


def test_pythons_did_you_mean_still_comes_from_real_names_only(sim):
    """Python's suggestion for a typo of a real name is as it was. Which
    versions' traceback module writes one differs (3.13 does; 3.12's
    format_exception_only does not), so the oracle is a plain class on this
    Python: the error carries the name and the object its suggestion is read
    from, and the text is the plain class's."""

    class Plain:
        __slots__ = ()

        def run(self) -> None: ...

    with pytest.raises(AttributeError) as plain:
        Plain().rn  # type: ignore[attr-defined]  # noqa: B018
    with pytest.raises(AttributeError) as err:
        sim.rn  # type: ignore[attr-defined]  # noqa: B018
    assert err.value.name == "rn" and err.value.obj is sim
    suggested = "Did you mean: 'run'?" in message(plain.value)
    assert ("Did you mean: 'run'?" in message(err.value)) == suggested
    # "simulat" is closer to the table's "simulate" than to anything real; the
    # table's names are kept out of dir(), so it is not suggested.
    with pytest.raises(AttributeError) as err:
        sim.simulat  # type: ignore[attr-defined]  # noqa: B018
    assert "'simulate'" not in message(err.value)


@pytest.mark.parametrize("owner", ["Model", "Simulator", "Result"])
@pytest.mark.parametrize(
    "probe",
    [
        "__deepcopy__",
        "__array__",
        "__array_interface__",
        "__array_struct__",
        "__fspath__",
        "__dataframe__",
        "_repr_html_",
        "_repr_mimebundle_",
        "_ipython_display_",
    ],
)
def test_protocol_probes_see_pythons_own_answer(owner, probe, model, sim, result):
    obj = instance(owner, model, sim, result)
    assert not hasattr(obj, probe)
    with pytest.raises(AttributeError) as err:
        getattr(obj, probe)
    assert str(err.value) == f"'{owner}' object has no attribute '{probe}'"


def test_copy_pickle_and_numpy_are_as_they_were(model, sim, result):
    for obj, what in ((model, "Model"), (sim, "Simulator"), (result, "Result")):
        with pytest.raises(TypeError, match=f"cannot pickle a {what}"):
            pickle.dumps(obj)
        with pytest.raises(TypeError, match=f"cannot pickle a {what}"):
            copy.deepcopy(obj)
    with pytest.raises(TypeError, match="cannot copy a Model"):
        copy.copy(model)
    # A copy taken before species is first read returns the same view too.
    unread = Simulator(Model.load(DECAY)).run(t_span=(0, 1), n_points=3)
    assert copy.copy(unread).species is unread.species
    as_array = np.asarray(result)
    assert as_array.shape == () and as_array.dtype == object and as_array[()] is result
    with pytest.raises(TypeError, match="not iterable"):
        iter(result)
    with pytest.raises(TypeError, match="not iterable"):
        "A_tot" in result  # noqa: B015
    assert not np.iterable(result)


def test_no_class_has_a_getattr_or_setattr_or_a_metaclass():
    """The success path stays free: CPython does not specialize a property read
    on a class with ``__getattr__`` (about 40 ns more per read), a metaclass
    ``__getattr__`` slows every class-attribute read, and a ``__setattr__``
    every write. The table's names are descriptors instead. No metaclass
    either, so a subclass can mix in an ABC."""
    for cls in (Model, Simulator, Result):
        assert "__getattr__" not in vars(cls)
        assert cls.__setattr__ is object.__setattr__
        assert type(cls) is type
    import abc

    class Mixed(Model, abc.ABC):
        pass

    assert issubclass(Mixed, Model)


# ─── Constructors given a path ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("given", "loader"),
    [
        (str(DECAY), "bngsim.Model.load(path)"),
        (DECAY, "bngsim.Model.load(path)"),
        ((DATA / "BIOMD0000000002.xml").read_text(), "bngsim.Model.from_sbml_string(text)"),
        ((DATA / "BIOMD0000000002.xml").read_bytes(), "bngsim.Model.from_sbml_string(text)"),
        (ANTIMONY, "bngsim.Model.from_antimony_string(text)"),
    ],
    ids=["str path", "Path", "SBML text", "SBML bytes", "Antimony text"],
)
def test_model_and_simulator_given_a_path_or_text_name_the_loader(given, loader):
    """``Model(path)`` and ``Simulator(path)`` failed inside bngsim, on
    ``'str' object has no attribute 'param_names'`` and ``'_want_output_sens'``."""
    with pytest.raises(TypeError) as err:
        Model(given)
    assert loader in str(err.value)
    with pytest.raises(TypeError) as err:
        Simulator(given)  # type: ignore[arg-type]
    assert f"bngsim.Simulator({loader})" in str(err.value)
    # And the loader named reads it.
    source = given.decode() if isinstance(given, bytes) else given
    built = {
        "bngsim.Model.load(path)": Model.load,
        "bngsim.Model.from_sbml_string(text)": Model.from_sbml_string,
        "bngsim.Model.from_antimony_string(text)": Model.from_antimony_string,
    }[loader](source)
    assert isinstance(built, Model)


def test_constructors_given_the_wrong_bngsim_object_name_the_right_one(model, sim, result):
    with pytest.raises(TypeError, match=r"Model\(\) is given a Model: model\.clone\(\)"):
        Model(model)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"bngsim\.Simulator\(sim\.model, method=\.\.\.\)"):
        Simulator(sim)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"given a Result, a run's output"):
        Simulator(result)  # type: ignore[arg-type]


def test_model_text_bngsim_reads_only_from_a_file_says_so():
    bngl = "begin model\nbegin parameters\n k 1\nend parameters\nend model\n"
    with pytest.raises(TypeError, match=r"BNGL text: use bngsim\.Model\.load\(path\) on a \.bngl"):
        Model(bngl)
    with pytest.raises(TypeError, match=r"\.net text: .* on a \.net file"):
        Simulator(DECAY.read_text())  # type: ignore[arg-type]


def test_a_string_that_is_no_path_is_not_called_one():
    with pytest.raises(TypeError, match=r"not a string: build the Model first"):
        Simulator("ode")  # type: ignore[arg-type]


@pytest.mark.parametrize("given", [None, 42, [1, 2]])
def test_model_and_simulator_given_something_else_say_what_they_take(given):
    with pytest.raises(TypeError, match=r"built by its loaders.*bngsim\.Model\.load\(path\)"):
        Model(given)
    with pytest.raises(TypeError, match=r"runs a bngsim\.Model, not a value of type"):
        Simulator(given)  # type: ignore[arg-type]


# ─── run() ───────────────────────────────────────────────────────────────────


def _corrected_call(err: pytest.ExceptionInfo) -> str:
    found = re.search(r"For this call: (sim\.run\(.*\))\.$", str(err.value))
    assert found, str(err.value)
    return found.group(1)


@pytest.mark.parametrize(
    ("call", "times"),
    [
        # RoadRunner's simulate(start, end, points): points counts the rows.
        ("sim.run(0, 10, 11)", np.linspace(0, 10, 11)),
        ("sim.run(end=10, points=11)", np.linspace(0, 10, 11)),
        ("sim.run(end=10, steps=10)", np.linspace(0, 10, 11)),
        # BioNetGen's n_steps counts the intervals: n_steps=10 is 11 rows.
        ("sim.run(t_end=10, n_steps=10)", np.linspace(0, 10, 11)),
        ("sim.run(t_start=2, t_end=10, n_steps=4)", np.linspace(2, 10, 5)),
        ("sim.run(t_span=(0, 10), n_steps=5)", np.linspace(0, 10, 6)),
        ("sim.run(t_end=10, n_points=6)", np.linspace(0, 10, 6)),
        ("sim.run(start=5, end=10, points=6)", np.linspace(5, 10, 6)),
        ("sim.run(0, 10, 11, ['A()'])", np.linspace(0, 10, 11)),  # selections dropped
        ("sim.run(0, 10, 11.0)", np.linspace(0, 10, 11)),  # a whole float is a count
        ("sim.run(t_end=10, n_steps=10.0)", np.linspace(0, 10, 11)),
        # SciPy's t_eval, and the obvious guess.
        ("sim.run(t_eval=[0, 1, 2.5])", np.array([0, 1, 2.5])),
        ("sim.run(times=[0, 1, 2.5])", np.array([0, 1, 2.5])),
        ("sim.run(times=np.array([0.0, 1.0, 2.5]))", np.array([0, 1, 2.5])),
    ],
)
def test_a_foreign_run_call_names_the_call_that_runs(fresh, call, times):
    """The corrected call the error gives is run, and gives the grid the
    foreign call meant: the oracle is each tool's own definition of its
    arguments, off by one where n_steps and steps count intervals."""
    with pytest.raises(TypeError) as err:
        eval(call, {"sim": fresh, "np": np})
    fixed = _corrected_call(err)
    got = eval(fixed, {"sim": fresh}).time
    np.testing.assert_allclose(got, times, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "says"),
    [
        ({"method": "ssa"}, r"Use bngsim\.Simulator\(model, method=\.\.\.\)"),
        ({"integrator": "gillespie"}, r"Use bngsim\.Simulator\(model, method=\.\.\.\)"),
        ({"selections": ["A"]}, r"result\.species\[name\]"),
        ({"y0": [1, 2]}, r"model\.set_concentration\(name, value\)"),
        ({"sample_time": [0, 1]}, r"Did you mean 'sample_times'\?"),
    ],
)
def test_a_run_keyword_with_no_one_to_one_translation_still_names_the_call(sim, kwargs, says):
    with pytest.raises(TypeError, match=says) as err:
        sim.run(**kwargs)
    assert "For this call" not in str(err.value)


@pytest.mark.parametrize(
    "call",
    [
        "sim.run(t_end=10, n_steps=10.5)",  # not a count: no corrected call
        "sim.run(t_span=(0, 10), t_end=5)",  # two ends
        "sim.run(n_points=5, n_steps=10)",  # two counts
    ],
)
def test_a_run_call_that_does_not_translate_gets_no_corrected_call(sim, call):
    with pytest.raises(TypeError) as err:
        eval(call, {"sim": sim})
    assert "For this call" not in str(err.value)


def test_a_number_for_start_names_t_span(sim):
    """run(start=...) names the state a run starts from (issue #1701); a number
    there is RoadRunner's start time, which bngsim takes in t_span."""
    with pytest.raises(ValueError, match=r"A start time goes in t_span: t_span=\(5, t_end\)"):
        sim.run(start=5)  # type: ignore[arg-type]


def test_an_unknown_run_keyword_reads_as_pythons(sim):
    with pytest.raises(TypeError) as err:
        sim.run(not_a_keyword=1)  # type: ignore[call-arg]
    assert str(err.value) == "Simulator.run() got an unexpected keyword argument 'not_a_keyword'"


@pytest.mark.parametrize(
    ("args", "says"),
    [
        ((10,), r"got the number 10: a run from 0 to 10 is t_span=\(0, 10\)"),
        ((0, 10), r"got the number 0: a run from 0 to 10 is t_span=\(0, 10\)"),
        ((5, 10), r"got the number 5: a run from 5 to 10 is t_span=\(5, 10\)"),
        # n_points' default, 101, is no end the caller wrote.
        ((0,), r"got the number 0\.$"),
    ],
)
def test_a_number_for_t_span_names_the_tuple(sim, args, says):
    with pytest.raises(TypeError, match=says):
        sim.run(*args)


def test_run_shows_its_own_signature_and_still_runs(fresh):
    params = inspect.signature(Simulator.run).parameters
    kinds = {p.kind for p in params.values()}
    assert inspect.Parameter.VAR_POSITIONAL not in kinds
    assert inspect.Parameter.VAR_KEYWORD not in kinds
    assert list(params)[:3] == ["self", "t_span", "n_points"]
    r = fresh.run((0, 10), 11)
    np.testing.assert_allclose(r.time, np.linspace(0, 10, 11), rtol=0, atol=1e-12)


def test_a_warning_from_run_names_the_callers_line():
    """run() checks its own arguments, rather than through a wrapper, so a
    warning it raises with a fixed stacklevel still names the line that called
    it, and each such line keeps its own once-per-location slot."""
    m = Model.from_antimony_string("model x\n  A -> B; k*A\n  k = 0.1; A = 10.5; B = 0\nend")
    ssa = Simulator(m, method="ssa")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ssa.run(t_span=(0, 1), n_points=3, seed=1)
    assert caught, "the non-integer count warns"
    assert {Path(w.filename).name for w in caught} == {Path(__file__).name}


# ─── The other run calls ─────────────────────────────────────────────────────


_GRID = [0, 2, 4, 6, 8, 10]  # 6 points over (0, 10): BioNetGen's n_steps=5


@pytest.mark.parametrize(
    ("method", "call", "check"),
    [
        (
            "ode",
            "sim.run_batch(t_end=10, n_steps=5, params=[{'k1': 0.05}, {'k1': 0.5}])",
            lambda out: [r.time.tolist() for r in out] == [_GRID, _GRID],
        ),
        (
            "ode",
            "sim.run_batch(0, 10, 6, params=[{'k1': 0.05}])",
            lambda out: out[0].time.tolist() == _GRID,
        ),
        (
            "ssa",
            "sim.run_replicates(2, 0, 10, 6, seed=1)",
            lambda out: len(out) == 2 and out[0].time.tolist() == _GRID,
        ),
        (
            "ssa",
            "sim.run_replicates(2, t_end=10, n_steps=5, seed=1)",
            lambda out: len(out) == 2 and out[1].time.tolist() == _GRID,
        ),
        (
            "ode",
            "sim.parameter_scan('k1', [0.05, 0.5], t_end=10, n_steps=5)",
            lambda out: len(out) == 2 and out[0].time.tolist() == _GRID,
        ),
        (
            "ode",
            "sim.run_until(5, n_steps=10)",
            lambda out: out.time.tolist() == np.linspace(0, 5, 11).tolist(),
        ),
    ],
)
def test_a_foreign_keyword_of_another_run_call_names_the_call_that_runs(method, call, check):
    """run_batch, run_replicates, parameter_scan and run_until name the keyword
    and give the corrected call as run() does, and the call they give runs and
    gives the grid the foreign one meant."""
    sim = Simulator(Model.load(DECAY), method=method)
    with pytest.raises(TypeError) as err:
        eval(call, {"sim": sim})
    fixed = re.search(r"For this call: (sim\.\w+\(.*\))\.$", str(err.value))
    assert fixed, str(err.value)
    assert check(eval(fixed.group(1), {"sim": sim}))


@pytest.mark.parametrize(
    ("call", "says"),
    [
        ("sim.run_batch(sample_times=[0, 1])", r"run_batch\(\) writes a uniform grid"),
        ("sim.run_replicates(2, times=[0, 1])", r"run_replicates\(\) writes a uniform grid"),
        ("sim.parameter_scan('k1', [0.1], t_eval=[0, 1])", r"parameter_scan\(\) writes a uniform"),
        ("sim.run_until(5, sample_times=[1, 2])", r"run_until\(\) writes a uniform grid"),
        ("sim.run_until(5, t_start=0)", r"Use sim\.set_time\(t_start\)"),
        ("sim.parameter_scan('k1', [0.1], suffix='x')", r"bngsim writes no files from a scan"),
        ("sim.parameter_scan('k1', values=[0.1])", r"Use par_scan_vals=values"),
        ("sim.run_batch(method='ssa')", r"Use bngsim\.Simulator\(model, method=\.\.\.\)"),
        ("sim.run_until(5, 10)", r"takes 1 positional argument \(t\) but 2 were given\.$"),
        ("sim.run_until(5, t_end=10)", r"Use t=t_end\. run_until\(\) runs on to the one time t"),
        (
            "sim.parameter_scan('k1', values=[0.1], t_end=10)",
            r"For this call: sim\.parameter_scan\(parameter='k1', par_scan_vals=\[0\.1\], "
            r"t_span=\(0, 10\)\)\.$",
        ),
        (
            "sim.run_batch(t_end=10, params=[{'k1': 0.1 * i} for i in range(1, 20)])",
            r"For this call: sim\.run_batch\(t_span=\(0, 10\), params=params\)\.$",
        ),
    ],
)
def test_another_run_call_names_its_own_keywords(sim, call, says):
    with pytest.raises(TypeError, match=says):
        eval(call, {"sim": sim})


def test_run_batch_reads_a_fourth_positional_list_of_dicts_as_its_params():
    """run_batch(0, 100, 101, rows): the rows are params, not RoadRunner's
    selections, so the corrected call keeps them."""
    sim = Simulator(Model.load(DECAY))
    rows = [{"k1": 0.05}, {"k1": 0.5}]
    with pytest.raises(TypeError) as err:
        sim.run_batch(0, 10, 6, rows)  # type: ignore[misc]
    fixed = re.search(r"For this call: (sim\.run_batch\(.*\))\.$", str(err.value))
    assert fixed and "params=" in fixed.group(1), str(err.value)
    out = eval(fixed.group(1), {"sim": sim})
    assert len(out) == 2 and out[0].time.tolist() == [0, 2, 4, 6, 8, 10]


@pytest.mark.parametrize(
    "call",
    [
        "sim.run_batch(0, 10, 11, t_end=5, params=[{'k1': 0.1}])",  # two ends
        "sim.run_batch(t_end=5, end=10, params=[{'k1': 0.1}])",
        "sim.run(t_start=1, start=2, t_end=5)",  # two starts
        "sim.run_batch(0, 10, 11, ['A()'], params=[{'k1': 0.1}])",  # selections: run() only
    ],
)
def test_a_call_that_says_one_thing_twice_gets_no_corrected_call(sim, call):
    with pytest.raises(TypeError) as err:
        eval(call, {"sim": sim})
    assert "For this call" not in str(err.value)


def test_a_value_whose_repr_is_no_literal_is_named_not_printed(sim):
    """par_scan_vals=np.linspace(...) prints as array([...]), which pasted is a
    NameError: the corrected call names the argument instead."""
    with pytest.raises(TypeError) as err:
        sim.parameter_scan("k1", np.array([0.1, 0.2]), t_end=10)  # type: ignore[call-arg]
    assert "array(" not in str(err.value)
    assert "par_scan_vals=par_scan_vals" in str(err.value)


def test_a_shared_row_naming_a_keyword_the_method_lacks_is_not_offered(sim):
    """keyword_entry filters run()'s rows: tspan= says t_span= on run(), and
    on run_until, which takes no t_span, says nothing of it."""
    with pytest.raises(TypeError, match=r"Use t_span="):
        sim.run(tspan=(0, 5))  # type: ignore[call-arg]
    with pytest.raises(TypeError) as err:
        sim.run_until(5, tspan=(0, 5))  # type: ignore[call-arg]
    assert "t_span=" not in str(err.value)


@pytest.mark.parametrize(
    ("call", "says"),
    [
        ("sim.run_batch(y0=[1, 2])", r"run_batch\(\) starts every point from the initial"),
        ("sim.run_batch(start='initial')", r"run_batch\(\) starts every point from the initial"),
        (
            "sim.run_replicates(2, start='initial')",
            r"Every replicate starts from the model's state",
        ),
        ("sim.parameter_scan('k1', [0.1], start='initial')", r"Use reset_to=label"),
        ("sim.run_until(5, max_step=0.1)", r"run_until\(\) takes no max_step"),
        ("sim.run_until(5, timeout=1)", r"run_until\(\) takes no timeout"),
        ("sim.run_until(5, duration=3)", r"Use t=sim\.current_time \+ duration"),
        ("sim.bifurcate('k1', [0.1, 0.2], t_end=10)", r"Use t_span=\(t_start, t_end\)"),
        ("sim.compute_all_sensitivities(n_steps=10)", r"Use n_points=n_steps \+ 1"),
    ],
)
def test_each_run_call_says_what_its_own_start_and_limits_are(sim, call, says):
    with pytest.raises(TypeError, match=says):
        eval(call, {"sim": sim})


@pytest.mark.parametrize(("args", "kwargs"), [((10, 0, 100), {}), (("k1", [0.1]), {"t_span": 10})])
def test_a_number_for_t_span_in_another_call_says_nothing_of_positions(args, kwargs):
    """run_replicates takes n_replicates first and parameter_scan t_span by
    keyword, so the message says nothing of where n_points sits."""
    sim = Simulator(Model.load(DECAY), method="ssa")
    call = sim.run_replicates if len(args) == 3 else sim.parameter_scan
    with pytest.raises(TypeError) as err:
        call(*args, **kwargs)
    assert "t_span is the interval" in str(err.value)
    assert "positional" not in str(err.value)


def test_the_other_run_calls_show_their_own_signatures():
    for method in fn.RUN_CALLS:
        params = inspect.signature(getattr(Simulator, method)).parameters.values()
        kinds = {p.kind for p in params}
        assert inspect.Parameter.VAR_POSITIONAL not in kinds, method
        assert inspect.Parameter.VAR_KEYWORD not in kinds, method


# ─── ParameterError and ModelError ───────────────────────────────────────────


def test_a_parameter_name_that_is_a_species_says_so_and_the_call_works():
    m = Model.from_antimony_string(ANTIMONY)
    with pytest.raises(ParameterError) as err:
        m.set_param("A", 5)
    assert str(err.value).startswith("Parameter 'A' not found in model.")
    assert "'A' is a species" in str(err.value)
    assert "model.set_concentration('A', 5)" in str(err.value)
    m.set_concentration("A", 5)
    assert m.get_concentration("A") == 5


def test_a_parameter_name_close_to_one_lists_it():
    m = Model.from_antimony_string(ANTIMONY)
    with pytest.raises(ParameterError, match=r"Close matches: 'kf'"):
        m.get_param("kff")
    with pytest.raises(ParameterError, match=r"Close matches: 'kcat'"):
        m.set_param("k_cat", 1.0)
    # bngsim's own synthesized slots (_rateLaw..., _V0_...) are not offered.
    with pytest.raises(ParameterError) as err:
        m.get_param("rateLaw")
    assert "_rateLaw" not in str(err.value)


def test_an_observable_or_a_bngl_species_named_as_a_parameter():
    m = Model.load(DECAY)
    with pytest.raises(ParameterError, match=r"'A_tot' is an observable.*result\.observables"):
        m.set_param("A_tot", 1.0)
    # BNGL names a species A(); A finds it.
    with pytest.raises(ParameterError, match=r"The species 'A\(\)' is close.*set_concentration"):
        m.set_param("A", 1.0)
    with pytest.raises(ParameterError, match=r"Model\.param_names lists all 3"):
        m.get_param("zz")


def test_set_params_says_what_each_unknown_name_is():
    m = Model.from_antimony_string(ANTIMONY)
    with pytest.raises(ParameterError) as err:
        m.set_params({"A": 1.0, "kff": 2.0, "kf": 0.3})
    text = str(err.value)
    assert text.startswith("Unknown parameter(s): A, kff.")
    assert "'A' is a species" in text and "For 'kff', close matches: 'kf'" in text
    assert "Known:" in text
    assert m.get_param("kf") == 0.1  # atomic: nothing was written


def test_a_species_name_that_is_a_parameter_says_so():
    m = Model.from_antimony_string(ANTIMONY)
    with pytest.raises(ModelError) as err:
        m.set_concentration("kf", 2.0)
    assert str(err.value).startswith("Species 'kf' not found in model.")
    assert "model.set_param('kf', 2)" in str(err.value)
    with pytest.raises(ModelError, match=r"Close matches: 'B'"):
        m.get_concentration("BB")


# ─── result.species["A"] ─────────────────────────────────────────────────────


def test_species_by_name_is_that_column(result):
    idx = result.species_names.index("A()")
    column = result.species["A()"]
    np.testing.assert_array_equal(column, result.species[:, idx])
    assert type(column) is np.ndarray and np.shares_memory(column, result.species)
    assert result.species[-1, "A()"] == result.species[-1, idx]
    np.testing.assert_array_equal(result.species[2:5, "A()"], result.species[2:5, idx])
    np.testing.assert_array_equal(result.species[[0, 3], "B()"], result.species[[0, 3], 1])
    # The decay it holds: A(t) = 100 exp(-(k1 + k2) t).
    np.testing.assert_allclose(column, 100 * np.exp(-0.1 * result.time), rtol=1e-6)


def test_species_by_name_on_a_batch_selects_the_last_axis():
    batch = Simulator(Model.load(DECAY)).run_batch(
        t_span=(0, 10), n_points=6, params=[{"k1": 0.05}, {"k1": 0.5}], squeeze=True
    )
    idx = batch.species_names.index("A()")
    assert batch.species["A()"].shape == (2, 6)
    np.testing.assert_array_equal(batch.species["A()"], batch.species[..., idx])
    np.testing.assert_array_equal(batch.species[1, "A()"], batch.species[1, :, idx])


def test_species_by_name_takes_a_fixed_species_with_or_without_its_dollar():
    r = Simulator(Model.load(FIXED)).run(t_span=(0, 5), n_points=3)
    assert "A()" in r.species_names
    np.testing.assert_array_equal(r.species["$A()"], r.species["A()"])


def test_species_by_a_wrong_name_says_where_it_is(result):
    with pytest.raises(KeyError, match=r"'A_tot' is an observable: result\.observables"):
        result.species["A_tot"]
    with pytest.raises(KeyError, match=r"Close matches: 'A\(\)'"):
        result.species["A"]
    with pytest.raises(KeyError, match=r"'A\(\)' is a species: result\.species\['A\(\)'\]"):
        result.observables["A()"]
    with pytest.raises(KeyError, match=r"^\"Expression 'x' not found"):
        result.expressions["x"]


@pytest.mark.parametrize(
    "derive",
    [
        lambda s: s.T,
        lambda s: s.copy(),
        lambda s: np.sort(s, axis=1),
        lambda s: s.astype(np.float32),
        lambda s: copy.deepcopy(s),
    ],
    ids=["T", "copy", "sort", "astype", "deepcopy"],
)
def test_an_array_derived_from_species_takes_no_name(result, derive):
    """A derived array's columns need not be result.species's: np.sort(axis=1)
    moves them. So none carries a name, rather than a stale one (the trap
    issue #561 found in NamedArray)."""
    with pytest.raises(KeyError, match="derived from result.species"):
        derive(result.species)["A()"]


def test_a_bngl_species_named_like_its_observable_is_named(data_dir=DATA):
    """BNGL's ``Molecules S1 S1()`` makes S1 an observable and S1() the species:
    asked for the species S1, the error names S1(), not the observable."""
    m = Model.load(data_dir / "case_sensitivity.net")
    assert "S1" in m.observable_names and "S1()" in m.species_names
    with pytest.raises(ModelError, match=r"The species 'S1\(\)' is close"):
        m.set_concentration("S1", 5.0)
    r = Simulator(m).run(t_span=(0, 1), n_points=3)
    with pytest.raises(KeyError, match=r"Close matches: 'S1\(\)'"):
        r.species["S1"]


def test_set_params_with_a_key_that_is_no_string_still_says_which():
    m = Model.from_antimony_string(ANTIMONY)
    with pytest.raises(ParameterError, match=r"Unknown parameter\(s\): 1\."):
        m.set_params({1: 2.0})  # type: ignore[dict-item]


def test_species_written_by_name():
    r = Simulator(Model.load(DECAY)).run(t_span=(0, 1), n_points=3)
    before = r.species.copy()
    r.species["A()"] *= 2
    np.testing.assert_array_equal(r.species["A()"], 2 * before[:, 0])
    r.species[1:, "B()"] = 0.0
    np.testing.assert_array_equal(r.species[:, 1], [before[0, 1], 0.0, 0.0])


def test_species_written_in_place_keeps_its_names():
    r = Simulator(Model.load(DECAY)).run(t_span=(0, 1), n_points=3)
    before = r.species["B()"].copy()
    species = r.species
    species += 1  # numpy hands back the out= array, and this name keeps it
    assert species is r.species
    np.testing.assert_array_equal(species["B()"], before + 1)
    assert np.add(species, 0, out=species) is species
    # A copy too: numpy hands back the out= array the caller passed.
    c = species.copy()
    alias = c
    c += 1
    assert c is alias


def test_species_is_otherwise_the_same_numpy_array(result):
    species = result.species
    assert isinstance(species, np.ndarray)
    assert species is result.species
    assert repr(species) == repr(np.asarray(species))
    for out in (species * 2, np.log1p(species), species.sum(axis=0), species[:, 0], species[1:]):
        assert type(out) is np.ndarray
    # A reduction is a number, as on a plain array (numpy 1.x gave a subclass a
    # 0-d array here).
    for total in (species.sum(), species.max(), np.mean(species), species[:, 0].sum()):
        assert isinstance(total, float) and np.ndim(total) == 0
    assert [type(row) for row in species] == [np.ndarray] * len(species)
    back = pickle.loads(pickle.dumps(species))
    assert type(back) is np.ndarray
    np.testing.assert_array_equal(back, species)


def test_a_saved_result_takes_species_names_too(result, tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path / "r.h5"
    result.save(path)
    loaded = Result.load(path)
    np.testing.assert_array_equal(loaded.species["B()"], result.species["B()"])


# ─── result["A"] ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("key", "says"),
    [
        ("A_tot", r"'A_tot' is an observable: result\.observables\['A_tot'\]"),
        ("B()", r"'B\(\)' is a species: result\.species\['B\(\)'\]"),
        ("A", r"No column is named 'A'; close matches: 'A\(\)'"),
        (0, r"result\.species\[0\], result\.observables\[0\]"),
        (slice(None, 2), r"result\.species\[\.\.\.\], result\.observables\[\.\.\.\]"),
    ],
)
def test_result_indexed_directly_names_the_block(result, key, says):
    with pytest.raises(TypeError, match=says) as err:
        result[key]  # type: ignore[index]
    assert str(err.value).startswith("'Result' object is not subscriptable")
