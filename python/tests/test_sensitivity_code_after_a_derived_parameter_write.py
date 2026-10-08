"""The sensitivity code a Simulator holds, after a derived parameter is
overridden or re-attached (issue #708).

``k2 = 2*k1`` follows its expression until ``set_param("k2", 5.0)`` pins it, and
follows it again after a write of the expression's own value (issue #188). The
generated sensitivity code differs between the two: attached, the column of
``k1`` carries the chain rule through ``k2``; pinned, ``k1`` moves nothing. An
artifact built for one was kept across the write, by the Simulator that held
it, by a new Simulator on the model, by a clone, and by every row of a batch.
The trajectory was right and the columns were those of the other case.

The model is ``A -> B`` at ``k2*A*A`` from ``A(0) = 10``, so
``A(t) = 10/(1 + 10*k2*t)`` and ``dA/dk2 = -100*t/(1 + 10*k2*t)**2``.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest
from bngsim import _codegen

ANT = "species A, B; A = 10; B = 0; k1 = 1; k2 = 2*k1;\nR1: A -> B; k2*A*A\n"
ANT_F = "species A, B; A = 10; B = 0; k1 = 1; k2 = 2*k1; F := k2*A;\nR1: A -> B; k2*A*A\n"
STEADY = "species A; A = 0; k1 = 1; kd = 0.5; k2 = 2*k1;\nJ0: -> A; k2\nJ1: A -> ; kd*A\n"
CHAIN = "species C; C = 0; R0 = 1; Q = 3*R0; Rt = 2*Q\nJ0: -> C; Rt\n"
NET = """begin parameters
    1 k1 0.3
    2 k2 2*k1
end parameters
begin species
    1 A() 10.0
    2 B() 0.0
end species
begin reactions
    1 1 2 k2
end reactions
begin groups
    1 Aobs 1
end groups
"""
P = ["k1", "k2"]
RUN = {"t_span": (0.0, 1.0), "n_points": 11, "rtol": 1e-10, "atol": 1e-12}


def _dk2(k2: float) -> float:
    """dA/dk2 at t = 1, with k2 its own axis."""
    return -100.0 / (1.0 + 10.0 * k2) ** 2


ATTACHED = [2.0 * _dk2(2.0), _dk2(2.0)]  # [-0.453515, -0.226757]
PINNED_AT_5 = [0.0, _dk2(5.0)]  # [0, -0.038447]


def _model(text: str = ANT) -> bngsim.Model:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bngsim.Model.from_antimony_string(text)


def _sim(model: bngsim.Model, params=P, **kw) -> bngsim.Simulator:
    return bngsim.Simulator(model, method="ode", sensitivity_params=list(params), **kw)


def _columns(result, species: str = "A") -> np.ndarray:
    j = list(result.species_names).index(species)
    return np.asarray(result.sensitivities)[-1, j]


def _a(result) -> float:
    return float(np.asarray(result.species)[-1, list(result.species_names).index("A")])


def _pin(model: bngsim.Model, value: float = 5.0) -> None:
    model.reset()
    model.set_param("k2", value)
    model.reset()


def _close(got, want) -> None:
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9)


# ── One Simulator across the write ───────────────────────────────────────────


def test_the_simulator_that_was_built_before_an_override():
    """The fitting loop: one Simulator, a derived parameter pinned between two
    runs. dA/dk1 came back -0.0769, the chain rule through a ``k2`` that no
    longer follows ``k1``, for 0."""
    model = _model()
    sim = _sim(model)
    _close(_columns(sim.run(**RUN)), ATTACHED)
    _pin(model)
    result = sim.run(**RUN)
    assert _a(result) == pytest.approx(10.0 / 51.0, rel=1e-8)
    _close(_columns(result), PINNED_AT_5)


def test_the_simulator_that_was_built_while_it_was_pinned():
    """The other way round: built while ``k2`` was pinned, run after a write of
    ``2*k1``'s own value attached it again. dA/dk1 came back 0 for -0.4535."""
    model = _model()
    model.set_param("k2", 5.0)
    sim = _sim(model)
    _close(_columns(sim.run(**RUN)), PINNED_AT_5)
    _pin(model, 2.0)
    _close(_columns(sim.run(**RUN)), ATTACHED)


def test_back_and_forth_on_one_simulator():
    """Each run has the columns of the model as it is attached then."""
    model = _model()
    sim = _sim(model)
    for value, want in ((5.0, PINNED_AT_5), (2.0, ATTACHED), (5.0, PINNED_AT_5), (2.0, ATTACHED)):
        _pin(model, value)
        _close(_columns(sim.run(**RUN)), want)


def test_the_columns_upstream_of_a_pinned_parameter_in_a_chain():
    """``Rt = 2*Q``, ``Q = 3*R0``, and ``-> C`` at ``Rt``. With ``Rt`` pinned at
    5, C = 5*t and neither ``Q`` nor ``R0`` moves it. The columns came back
    [4, 12, 2] for [0, 0, 2]."""
    model = _model(CHAIN)
    sim = _sim(model, ["Q", "R0", "Rt"])
    times = [0.0, 1.0, 2.0]
    _close(np.asarray(sim.run(sample_times=times).sensitivities)[-1, 0], [4.0, 12.0, 2.0])
    model.reset()
    model.set_param("Rt", 5.0, force_override=True)
    result = sim.run(sample_times=times, rtol=1e-10, atol=1e-12)
    _close(np.asarray(result.sensitivities)[-1, 0], [0.0, 0.0, 2.0])


def test_a_reported_expression_s_columns_follow_it_too():
    """``F := k2*A`` is reported, and its columns are evaluated by the same
    artifact. Pinned at 5, dF/dk1 came back 0.00769 for 0."""
    model = _model(ANT_F)
    sim = _sim(model)
    attached = np.asarray(sim.run(**RUN).output_sensitivities("F"))[-1, 0]
    _close(attached, [20.0 / 21.0**2, 10.0 / 21.0**2])
    _pin(model)
    pinned = np.asarray(sim.run(**RUN).output_sensitivities("F"))[-1, 0]
    _close(pinned, [0.0, 10.0 / 51.0**2])


def test_a_net_model_s_derived_parameter(tmp_path):
    """The same for a ``.net`` parameter written as an expression of another.
    ``A -> B`` at ``k2``: A = 10*exp(-k2*t), and with ``k2`` pinned at 5 the
    column of ``k1`` is 0."""
    path = tmp_path / "derived.net"
    path.write_text(NET)
    model = bngsim.Model.from_net(str(path))
    sim = _sim(model)
    _close(_columns(sim.run(**RUN), "A()"), [-20.0 * np.exp(-0.6), -10.0 * np.exp(-0.6)])
    _pin(model)
    _close(_columns(sim.run(**RUN), "A()"), [0.0, -10.0 * np.exp(-5.0)])


# ── What a later Simulator takes from the model ──────────────────────────────


def test_a_new_simulator_on_the_model_after_an_override():
    """The model carries the artifact an earlier Simulator built, and a new
    Simulator took it as built. Same wrong columns, [-0.0769, -0.0384]."""
    model = _model()
    _sim(model).run(**RUN)
    _pin(model)
    _close(_columns(_sim(model).run(**RUN)), PINNED_AT_5)


def test_a_new_simulator_on_the_model_after_it_is_attached_again():
    model = _model()
    model.set_param("k2", 5.0)
    _sim(model).run(**RUN)
    _pin(model, 2.0)
    _close(_columns(_sim(model).run(**RUN)), ATTACHED)


def test_a_simulator_on_a_clone_taken_after_an_override():
    """``clone()`` copies the artifact with the model."""
    model = _model()
    _sim(model).run(**RUN)
    _pin(model)
    clone = model.clone()
    clone.reset()
    _close(_columns(_sim(clone).run(**RUN)), PINNED_AT_5)


def test_a_clone_overridden_after_it_was_taken():
    """The clone is pinned, the model it was cut from is not, and each has
    its own columns."""
    model = _model()
    _sim(model).run(**RUN)
    clone = model.clone()
    _pin(clone)
    _close(_columns(_sim(clone).run(**RUN)), PINNED_AT_5)
    model.reset()
    _close(_columns(_sim(model).run(**RUN)), ATTACHED)


def test_a_plain_simulator_built_in_between_does_not_cost_the_columns():
    """A plain Simulator on the model clears the flag the build reads to decide
    whether to emit sensitivity code. The rebuild after an override still
    emits it."""
    model = _model()
    sim = _sim(model)
    sim.run(**RUN)
    bngsim.Simulator(model, method="ode")
    _pin(model)
    result = sim.run(**RUN)
    assert sim.has_analytic_sens_rhs
    _close(_columns(result), PINNED_AT_5)


# ── The entry points that take their columns as an argument ──────────────────


@pytest.mark.parametrize("built_with_columns", [True, False])
def test_compute_all_sensitivities_after_an_override(built_with_columns):
    model = _model()
    sim = _sim(model) if built_with_columns else bngsim.Simulator(model, method="ode")
    _close(_columns(sim.compute_all_sensitivities(params=P, **RUN)), ATTACHED)
    _pin(model)
    _close(_columns(sim.compute_all_sensitivities(params=P, **RUN)), PINNED_AT_5)
    _pin(model, 2.0)
    _close(_columns(sim.compute_all_sensitivities(params=P, **RUN)), ATTACHED)


def test_the_steady_state_after_an_override():
    """``-> A`` at ``k2``, ``A ->`` at ``kd*A``: A* = k2/kd. Pinned, dA*/dk1 is
    0. It came back 4, the attached answer."""
    model = _model(STEADY)
    sim = bngsim.Simulator(model, method="ode")
    _close(np.asarray(sim.steady_state(sensitivity_params=P, tol=1e-12).sensitivity)[0], [4, 2])
    _pin(model)
    _close(np.asarray(sim.steady_state(sensitivity_params=P, tol=1e-12).sensitivity)[0], [0, 2])
    _pin(model, 2.0)
    _close(np.asarray(sim.steady_state(sensitivity_params=P, tol=1e-12).sensitivity)[0], [4, 2])


# ── A batch ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("squeeze", [False, True])
def test_a_batch_row_that_overrides_a_derived_parameter_is_refused(squeeze):
    """Every row runs on the one artifact. The row ``{"k2": 5.0}`` returned
    dA/dk1 = -0.0769 for 0."""
    sim = _sim(_model())
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#708") as refusal:
        sim.run_batch(params=[{"k1": 1.5}, {"k2": 5.0}], squeeze=squeeze, **RUN)
    assert "row 1" in str(refusal.value)
    assert "overrides the derived parameter(s) ['k2']" in str(refusal.value)


def test_a_batch_row_that_attaches_one_again_is_refused():
    """On a model with ``k2`` pinned, the row ``{"k2": 2.0}`` writes the value
    of ``2*k1`` and attaches it. It ran on the pinned code: dA/dk1 = 0 for
    -0.4535."""
    model = _model()
    model.set_param("k2", 5.0)
    sim = _sim(model)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#708") as refusal:
        sim.run_batch(params=[{"k2": 2.0}], **RUN)
    assert "attaches it" in str(refusal.value)


def test_a_batch_on_a_model_pinned_after_the_simulator_was_built():
    """The write is made on the model, as the refusal says to make it, and the
    rows then write other values of the pinned parameter. The batch runs on
    the code for the model as it is: [-0.2081, -0.1041] came back for
    [0, -0.1041]."""
    model = _model()
    sim = _sim(model)
    sim.run(**RUN)
    _pin(model)
    rows = sim.run_batch(params=[{"k2": 3.0}, {"k2": 5.0}], **RUN)
    _close(_columns(rows[0]), [0.0, _dk2(3.0)])
    _close(_columns(rows[1]), PINNED_AT_5)


def test_a_batch_row_that_writes_another_parameter():
    """Control. A row that leaves the derived parameters attached as they were
    runs as it did: at ``k1 = 2`` the rate constant is 4."""
    rows = _sim(_model()).run_batch(params=[{"k1": 2.0}], **RUN)
    _close(_columns(rows[0]), [2.0 * _dk2(4.0), _dk2(4.0)])


def test_a_batch_that_asks_for_initial_conditions_only():
    """Control. The chain rule is in the parameter columns. A row that pins
    ``k2`` in a batch of initial-condition columns runs, and
    dA/dA(0) = 1/(1 + 10*k2)**2."""
    sim = bngsim.Simulator(_model(), method="ode", sensitivity_ic=["A"])
    rows = sim.run_batch(params=[{"k2": 5.0}, {"k1": 2.0}], **RUN)
    for row, k2 in zip(rows, (5.0, 4.0), strict=True):
        j = list(row.species_names).index("A")
        got = np.asarray(row.sensitivities_ic)[-1, j, 0]
        assert got == pytest.approx(1.0 / (1.0 + 10.0 * k2) ** 2, rel=1e-6)


# ── What does not change ─────────────────────────────────────────────────────


def test_a_plain_compiled_run_reads_the_pinned_value():
    """Control. The state right-hand side reads a derived parameter's value
    where the model keeps it, so a plain artifact serves either attachment:
    one Simulator across the write, and the rows of a batch."""
    model = _model()
    sim = bngsim.Simulator(model, method="ode", codegen=True)
    assert _a(sim.run(**RUN)) == pytest.approx(10.0 / 21.0, rel=1e-8)
    _pin(model)
    assert _a(sim.run(**RUN)) == pytest.approx(10.0 / 51.0, rel=1e-8)
    rows = sim.run_batch(params=[{"k2": 2.0}, {"k2": 7.0}], **RUN)
    assert _a(rows[0]) == pytest.approx(10.0 / 21.0, rel=1e-8)
    assert _a(rows[1]) == pytest.approx(10.0 / 71.0, rel=1e-8)


def _count_builds(monkeypatch) -> list[int]:
    """Count the calls that build or look up an artifact for a model."""
    calls = [0]
    for name in ("prepare_model_codegen", "prepare_model_codegen_source"):
        real = getattr(_codegen, name)

        def counted(model, _real=real):
            calls[0] += 1
            return _real(model)

        monkeypatch.setattr(_codegen, name, counted)
    return calls


def test_a_write_to_a_primary_builds_nothing(monkeypatch):
    """Control. A fit moves rate constants. No derived parameter is pinned or
    attached by that, and the artifact is the one the Simulator was built
    with."""
    model = _model()
    sim = _sim(model)
    calls = _count_builds(monkeypatch)
    for k1 in (1.5, 0.7, 1.0):
        model.reset()
        model.set_param("k1", k1)
        _close(_columns(sim.run(**RUN)), [2.0 * _dk2(2.0 * k1), _dk2(2.0 * k1)])
    assert calls[0] == 0


def test_the_code_is_built_once_for_each_change_of_attachment(monkeypatch):
    """A second run on the model as it stands builds nothing, and a later
    value of the pinned parameter builds nothing either."""
    model = _model()
    sim = _sim(model)
    calls = _count_builds(monkeypatch)
    _pin(model)
    sim.run(**RUN)
    assert calls[0] == 1
    model.reset()
    sim.run(**RUN)
    _pin(model, 3.0)
    _close(_columns(sim.run(**RUN)), [0.0, _dk2(3.0)])
    assert calls[0] == 1
    _pin(model, 2.0)
    sim.run(**RUN)
    assert calls[0] == 2


def test_a_rebuild_that_fails_is_a_refusal(monkeypatch):
    """Where the code cannot be built for the model as it is now attached, the
    run is refused. It is not run on the code for the other attachment."""
    model = _model()
    sim = _sim(model)
    sim.run(**RUN)
    _pin(model)
    for name in ("prepare_model_codegen", "prepare_model_codegen_source"):
        monkeypatch.setattr(_codegen, name, lambda model: None)
    with pytest.raises(bngsim.SensitivityUnsupportedError):
        sim.run(**RUN)
    # And again: the refusal does not leave the Simulator with no compiled code
    # and nothing to compare, to run the next request on the interpreter.
    model.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError):
        sim.run(**RUN)
    with pytest.raises(bngsim.SensitivityUnsupportedError):
        sim.run_batch(params=[{"k1": 2.0}], **RUN)


def test_a_failed_regeneration_does_not_serve_code_for_another_attachment(monkeypatch):
    """``compute_all_sensitivities`` on a plain Simulator drops its artifact to
    build one with sensitivity code, and puts the old one back if that fails,
    for the call to go on with. Where the old one was built before a derived
    parameter was pinned the call is refused, each time it is made, and the
    plain run keeps its compiled code."""
    model = _model()
    sim = bngsim.Simulator(model, method="ode", codegen=True)
    sim.run(**RUN)
    before = (sim._codegen_so_path, sim._codegen_c_source)
    _pin(model)
    for name in ("prepare_model_codegen", "prepare_model_codegen_source"):
        monkeypatch.setattr(_codegen, name, lambda model: None)
    for _ in range(2):
        model.reset()
        with pytest.raises(bngsim.SensitivityUnsupportedError):
            sim.compute_all_sensitivities(params=P, **RUN)
    assert (sim._codegen_so_path, sim._codegen_c_source) == before
    model.reset()
    assert _a(sim.run(**RUN)) == pytest.approx(10.0 / 51.0, rel=1e-8)


def test_a_failed_regeneration_puts_back_the_code_for_this_attachment(monkeypatch):
    """Control. With nothing pinned since, the old artifact is put back, as it
    was, and the call goes on to its own answer."""
    model = _model()
    sim = bngsim.Simulator(model, method="ode", codegen=True)
    sim.run(**RUN)
    before = (sim._codegen_so_path, sim._codegen_c_source)
    for name in ("prepare_model_codegen", "prepare_model_codegen_source"):
        monkeypatch.setattr(_codegen, name, lambda model: None)
    model.reset()
    _close(_columns(sim.compute_all_sensitivities(params=P, **RUN)), ATTACHED)
    assert (sim._codegen_so_path, sim._codegen_c_source) == before


# ── What is remembered about the artifact is asked again ─────────────────────

ANT_FLOOR = "species A, B; A = 10; B = 0; k1 = 1; k2 = floor(k1) + 2;\nR1: A -> B; k2*A*A\n"
ANT_ABS = "species A, B; A = 10; B = 0; k1 = 1; k2 = abs(k1 - 3); F := k2*A;\nR1: A -> B; k2*A*A\n"


def test_whether_the_code_holds_the_analytic_columns_is_asked_of_the_new_code():
    """``k2 = floor(k1) + 2`` cannot be differentiated, so while it is attached
    there is no analytic sensitivity right-hand side. Pinned, there is one, and
    the Simulator built before the pin says so and runs on it."""
    model = _model(ANT_FLOOR)
    sim = _sim(model)
    assert not sim.has_analytic_sens_rhs
    _pin(model)
    result = sim.run(**RUN)
    assert sim.has_analytic_sens_rhs
    _close(_columns(result), PINNED_AT_5)


def test_a_new_simulator_is_not_told_what_another_attachment_s_code_holds():
    """Built while ``k2 = floor(k1) + 2`` was pinned, the model's artifact has
    the analytic columns. With ``k2`` attached again there are none to have,
    and a new Simulator says so before it has run anything."""
    model = _model(ANT_FLOOR)
    model.set_param("k2", 5.0)
    assert _sim(model).has_analytic_sens_rhs
    _pin(model, 3.0)
    assert not _sim(model).has_analytic_sens_rhs


def test_an_expression_that_can_be_differentiated_once_pinned():
    """``F := k2*A`` over ``k2 = abs(k1 - 3)`` has no output sensitivity while
    ``k2`` is attached. A Simulator built then went on saying so after the pin,
    where dF/d(k1, k2) is [0, 10/51**2]."""
    model = _model(ANT_ABS)
    sim = _sim(model)
    with pytest.raises(ValueError, match="derived parameter 'k2'"):
        sim.run(**RUN).output_sensitivities("F")
    _pin(model)
    _close(np.asarray(sim.run(**RUN).output_sensitivities("F"))[-1, 0], [0.0, 10.0 / 51.0**2])


def test_an_expression_that_cannot_be_differentiated_once_attached():
    """The other way round. Built while ``k2`` was pinned, the Simulator gave
    dF/dk1 = 0 after ``k2`` was attached again, for -10/21**2, and dA/dk1 = 0
    for 0.2268. The columns are right now, and the expression is refused, as
    it is on a Simulator built then."""
    model = _model(ANT_ABS)
    model.set_param("k2", 5.0)
    sim = _sim(model)
    _close(np.asarray(sim.run(**RUN).output_sensitivities("F"))[-1, 0], [0.0, 10.0 / 51.0**2])
    _pin(model, 2.0)
    result = sim.run(**RUN)
    _close(_columns(result), [-_dk2(2.0), _dk2(2.0)])
    with pytest.raises(ValueError, match="derived parameter 'k2'"):
        result.output_sensitivities("F")


# ── A Simulator that took the model's artifact ───────────────────────────────


def test_a_second_simulator_that_took_the_first_one_s_artifact():
    """Two Simulators on one model, both built before the override. The second
    took the artifact the first left on the model, and what it was built for
    came with it."""
    model = _model()
    first = _sim(model)
    second = _sim(model)
    assert second._codegen_so_path == first._codegen_so_path
    assert second._codegen_c_source == first._codegen_c_source
    _pin(model)
    _close(_columns(second.run(**RUN)), PINNED_AT_5)
    model.reset()
    _close(_columns(first.run(**RUN)), PINNED_AT_5)


def test_an_artifact_built_without_being_asked_for(monkeypatch):
    """A model at the size threshold is compiled without ``codegen=True``, and
    that artifact is on the model before the Simulator takes it. It is put
    back after a failed regeneration only for the attachment it was built
    for."""
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
    model = _model()
    sim = bngsim.Simulator(model, method="ode")
    assert sim._codegen_so_path or sim._codegen_c_source
    sim.run(**RUN)
    _pin(model)
    for name in ("prepare_model_codegen", "prepare_model_codegen_source"):
        monkeypatch.setattr(_codegen, name, lambda model: None)
    with pytest.raises(bngsim.SensitivityUnsupportedError):
        sim.compute_all_sensitivities(params=P, **RUN)
