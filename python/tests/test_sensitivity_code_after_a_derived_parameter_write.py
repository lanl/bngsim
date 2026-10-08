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


def test_a_build_that_failed_is_not_asked_for_again_while_nothing_has_changed(monkeypatch):
    """Control. A plain compiled Simulator whose sensitivity build fails goes
    on with the artifact it has, each time it is asked for columns, and pays
    for the failed build once."""
    model = _model()
    sim = bngsim.Simulator(model, method="ode", codegen=True)
    sim.run(**RUN)
    calls = [0]

    def fails(model):
        calls[0] += 1

    for name in ("prepare_model_codegen", "prepare_model_codegen_source"):
        monkeypatch.setattr(_codegen, name, fails)
    for _ in range(3):
        model.reset()
        _close(_columns(sim.compute_all_sensitivities(params=P, **RUN)), ATTACHED)
    assert calls[0] == 1


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
    # Asked before the next run as well as after it.
    assert sim.has_analytic_sens_rhs
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


# ── What the review found ────────────────────────────────────────────────────


def test_a_pin_at_the_value_the_expression_gives_moves_no_value():
    """``force_override`` pins ``k2`` at 2, the value ``2*k1`` gives. No
    parameter has moved, and ``k1`` no longer reaches the rate."""
    model = _model()
    sim = _sim(model)
    _close(_columns(sim.run(**RUN)), ATTACHED)
    model.reset()
    model.set_param("k2", 2.0, force_override=True)
    model.reset()
    _close(_columns(sim.run(**RUN)), [0.0, _dk2(2.0)])


ON_THE_QUOTIENT = (
    "species A, B; A = 10; B = 0; k1 = 1; k2 = 2*k1; z = 1.5; kz = floor(z) + 1; kb = 1;\n"
    "R1: A -> B; kz*piecewise(kb, k2 > 4.999, 0)*A\n"
)


def test_what_is_kept_about_a_run_on_the_quotient_goes_with_the_attachment():
    """``set_params({"k2": 5.0, "k1": 2.5})`` names ``k2`` first, so the first
    such write pins it and the second, an identity by then, attaches it again:
    two attachments with every value the same. ``kz = floor(z) + 1`` keeps the
    model on the difference quotient, where what was decided for the pinned
    model was kept by the values alone. Attached, ``k1`` moves ``k2`` across
    ``k2 > 4.999`` and the run is refused (issue #938); it ran, and returned
    dA/dk1 = -0.378 for 0."""
    theta = {"k2": 5.0, "k1": 2.5}
    loose = {"t_span": (0.0, 1.0), "n_points": 11, "rtol": 1e-4, "atol": 1e-7}
    model = _model(ON_THE_QUOTIENT)
    model.set_params(theta)
    model.reset()
    index = list(model.param_names).index("k2")
    assert not model._core.param_is_expression[index]
    sim = _sim(model, ["k1", "kb"])
    pinned = _columns(sim.run(**loose))
    np.testing.assert_allclose(pinned, [0.0, -20.0 * np.exp(-2.0)], rtol=1e-3, atol=1e-6)
    model.reset()
    model.set_params(theta)
    model.reset()
    assert model._core.param_is_expression[index]
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938"):
        sim.run(**loose)


@pytest.mark.parametrize("between", ["ode", "ssa"])
def test_what_an_expression_can_be_differentiated_for_whichever_way_the_code_was_replaced(between):
    """A Simulator built on the model in between sends the next
    ``compute_all_sensitivities`` down its other rebuild. What ``F`` can be
    differentiated for was kept across that one: built pinned and then
    attached it returned [nan, nan] for ``F``, where a new Simulator refuses
    it by name, and built attached and then pinned it refused where
    dF/d(k1, k2) is [0, 10/51**2]."""
    model = _model(ANT_ABS)
    model.set_param("k2", 5.0)
    sim = bngsim.Simulator(model, method="ode")
    sim.compute_all_sensitivities(params=P, **RUN)
    bngsim.Simulator(model, method=between)
    _pin(model, 2.0)
    with pytest.raises(ValueError, match="derived parameter 'k2'"):
        sim.compute_all_sensitivities(params=P, **RUN).output_sensitivities("F")

    model = _model(ANT_ABS)
    sim = bngsim.Simulator(model, method="ode")
    sim.compute_all_sensitivities(params=P, **RUN)
    bngsim.Simulator(model, method=between)
    _pin(model)
    result = sim.compute_all_sensitivities(params=P, **RUN)
    _close(np.asarray(result.output_sensitivities("F"))[-1, 0], [0.0, 10.0 / 51.0**2])


# ── A plain Simulator and a sensitivity Simulator on one model ───────────────


def test_a_plain_compiled_simulator_asked_for_columns_after_a_sensitivity_one_was_built():
    """A plain compiled Simulator's artifact has no sensitivity code. Whether to
    build one that has was read off a flag on the model, which a sensitivity
    Simulator built there in between had set: the plain artifact was kept, the
    columns ran on the difference quotient with nothing said, and ``F`` had no
    output sensitivity."""
    model = _model(ANT_F)
    plain = bngsim.Simulator(model, method="ode", codegen=True)
    _sim(model)
    model.reset()
    result = plain.compute_all_sensitivities(params=P, **RUN)
    assert plain._codegen_provides_sens_rhs()
    _close(_columns(result), ATTACHED)
    _close(np.asarray(result.output_sensitivities("F"))[-1, 0], [20.0 / 21.0**2, 10.0 / 21.0**2])


def test_a_rebuild_on_one_simulator_does_not_cost_another_its_columns():
    """The rebuild after a pin sets that flag for its own build. The plain
    compiled Simulator on the same model still builds the code it needs when
    it is asked for columns."""
    model = _model(ANT_F)
    sim = _sim(model)
    plain = bngsim.Simulator(model, method="ode", codegen=True)
    _pin(model)
    _close(_columns(sim.run(**RUN)), PINNED_AT_5)
    _pin(model, 2.0)
    result = plain.compute_all_sensitivities(params=P, **RUN)
    assert plain._codegen_provides_sens_rhs()
    _close(np.asarray(result.output_sensitivities("F"))[-1, 0], [20.0 / 21.0**2, 10.0 / 21.0**2])


def test_a_sensitivity_simulator_does_not_take_a_plain_artifact_after_a_failed_rebuild(
    monkeypatch,
):
    """A rebuild that fails leaves the flag set and the model with the plain
    artifact another Simulator put there. A sensitivity Simulator built then
    builds its own."""
    model = _model()
    sim = _sim(model)
    bngsim.Simulator(model, method="ode", codegen=True)
    _pin(model)
    with monkeypatch.context() as broken:
        for name in ("prepare_model_codegen", "prepare_model_codegen_source"):
            broken.setattr(_codegen, name, lambda model: None)
        with pytest.raises(bngsim.SensitivityUnsupportedError):
            sim.run(**RUN)
    model.reset()
    later = _sim(model)
    assert later.has_analytic_sens_rhs
    _close(_columns(later.run(**RUN)), PINNED_AT_5)


def test_a_plain_artifact_is_not_taken_for_a_sensitivity_one_by_the_flag(monkeypatch):
    """A plain compiled Simulator asked for columns sets the flag for the
    build it then makes. Where that build fails it keeps its plain artifact
    and goes on with the difference quotient, and the model is left with the
    flag set and the plain artifact. A sensitivity Simulator built then took
    it, and ran on the difference quotient with nothing said."""
    model = _model()
    plain = bngsim.Simulator(model, method="ode", codegen=True)
    with monkeypatch.context() as broken:
        for name in ("prepare_model_codegen", "prepare_model_codegen_source"):
            broken.setattr(_codegen, name, lambda model: None)
        plain.compute_all_sensitivities(params=P, **RUN)
    model.reset()
    sim = _sim(model)
    assert sim._codegen_provides_sens_rhs()
    _close(_columns(sim.run(**RUN)), ATTACHED)


# ── A condition that is read otherwise at the new values ─────────────────────

GATED = (
    "species A, B; A = 10; B = 0; k = 1; E = 0.25;\n"
    "R1: A -> B; piecewise(k, time >= sqrt(E), 0)*A\n"
)
TIGHT = {"t_span": (0.0, 1.0), "n_points": 11, "rtol": 1e-9, "atol": 1e-12}


def _gated(e: float) -> list[float]:
    """dA/d(k, E) at t = 1 for A = 10*exp(-k*(1 - sqrt(E))) at k = 1."""
    a = 10.0 * np.exp(-(1.0 - np.sqrt(e)))
    return [-(1.0 - np.sqrt(e)) * a, a / (2.0 * np.sqrt(e))]


def test_a_write_that_changes_how_a_condition_is_read_is_followed_by_a_rebuild():
    """The switch is at ``sqrt(E)``. At ``E = 0`` it does not resolve to a time
    a column can be carried across, a model loaded there has no analytic
    right-hand side and the run is refused. On the artifact built at 0.25 the
    run went ahead and returned dA/dE = 0, where the derivative is unbounded.
    Back at 0.25 the Simulator runs again."""
    model = _model(GATED)
    sim = _sim(model, ["k", "E"])
    _close(_columns(sim.run(**TIGHT)), _gated(0.25))
    model.reset()
    model.set_param("E", 0.0)
    model.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="crossing time"):
        sim.run(**TIGHT)
    assert not sim.has_analytic_sens_rhs
    model.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="crossing time"):
        sim.compute_all_sensitivities(params=["k", "E"], **TIGHT)
    clone = model.clone()
    clone.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="crossing time"):
        _sim(clone, ["k", "E"]).run(**TIGHT)
    model.reset()
    model.set_param("E", 0.25)
    model.reset()
    _close(_columns(sim.run(**TIGHT)), _gated(0.25))
    assert sim.has_analytic_sens_rhs


def test_every_column_at_once_is_asked_of_the_conditions_as_they_are():
    """``compute_all_sensitivities`` runs its columns on clones. Asked first
    at ``E = 0``, of a Simulator built at 0.25, it is refused as a run is."""
    model = _model(GATED)
    sim = _sim(model, ["k", "E"])
    _close(_columns(sim.compute_all_sensitivities(params=["k", "E"], **TIGHT)), _gated(0.25))
    model.reset()
    model.set_param("E", 0.0)
    model.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="crossing time"):
        sim.compute_all_sensitivities(params=["k", "E"], **TIGHT)


SCHEDULE = (
    "species A, B; A = 10; B = 0; k = 1; P = 0.4; E = 0.0625;\n"
    "R1: A -> B; piecewise(k, time - P*floor(time/P) >= sqrt(E), 0)*A\n"
)


def _scheduled(e: float) -> list[float]:
    """dA/d(k, E) at t = 1. The law is on for the last ``0.4 - sqrt(E)`` of
    each period of 0.4, twice by t = 1 (the third window opens after it), so
    A = 10*exp(-2*k*(0.4 - sqrt(E)))."""
    on = 2.0 * (0.4 - np.sqrt(e))
    a = 10.0 * np.exp(-on)
    return [-on * a, a / np.sqrt(e)]


def test_a_repeating_schedule_whose_duty_stops_resolving_is_followed_by_a_rebuild():
    """The same for a schedule, which the pass reads on another path: on for
    ``(t mod P) >= sqrt(E)``. At ``E = 0`` it is always on and dA/dE is
    unbounded; a model loaded there is refused. On the artifact built at 0.0625
    every one of these returned dA/dE = 0."""
    model = _model(SCHEDULE)
    sim = _sim(model, ["k", "E"])
    _close(_columns(sim.run(**TIGHT)), _scheduled(0.0625))
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#708") as refusal:
        sim.run_batch(params=[{"E": 0.0}], **TIGHT)
    assert "row 0" in str(refusal.value)
    model.reset()
    model.set_param("E", 0.0)
    model.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="crossing time"):
        sim.run(**TIGHT)
    assert not sim.has_analytic_sens_rhs
    model.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="crossing time"):
        sim.compute_all_sensitivities(params=["k", "E"], **TIGHT)
    model.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="crossing time"):
        _sim(model, ["k", "E"]).run(**TIGHT)
    clone = model.clone()
    clone.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="crossing time"):
        _sim(clone, ["k", "E"]).run(**TIGHT)
    model.reset()
    model.set_param("E", 0.0625)
    model.reset()
    _close(_columns(sim.run(**TIGHT)), _scheduled(0.0625))
    assert sim.has_analytic_sens_rhs


def test_a_batch_after_a_write_made_on_the_model_is_asked_what_a_run_is():
    """The write is made on the model, as the row refusal says to make it, and
    the batch's rows change nothing. It is refused as a ``run()`` is, for the
    model, and not as a row that moved the parameter. It returned dA/dE = 0."""
    model = _model(GATED)
    sim = _sim(model, ["k", "E"])
    model.set_param("E", 0.0)
    model.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="crossing time") as refusal:
        sim.run_batch(params=[{}], **TIGHT)
    assert "row 0" not in str(refusal.value)
    model.set_param("E", 0.25)
    model.reset()
    _close(_columns(sim.run_batch(params=[{}], **TIGHT)[0]), _gated(0.25))


def test_a_batch_asks_its_own_model_again_only_after_something_has_changed(monkeypatch):
    """The pass on the batch's own model is made once for the values, the
    attachment, the columns and the window it was made for. A batch of one row
    in a loop paid it each time. A write in between is seen: at ``E = 0`` the
    next batch is refused, and back at 0.25 it runs."""
    from bngsim import _switch_sensitivity

    model = _model(GATED)
    sim = _sim(model, ["k", "E"])
    own = []
    real = _switch_sensitivity.compute_switch_time_sens

    def counted(core, *args, **kwargs):
        if core is model._core:
            own.append(1)
        return real(core, *args, **kwargs)

    monkeypatch.setattr(_switch_sensitivity, "compute_switch_time_sens", counted)
    for _ in range(3):
        _close(_columns(sim.run_batch(params=[{}], **TIGHT)[0]), _gated(0.25))
    assert len(own) == 1
    sim.run_batch(params=[{}], t_span=(0.0, 2.0), n_points=11, rtol=1e-9, atol=1e-12)
    assert len(own) == 2
    model.set_param("k", 2.0)
    model.reset()
    sim.run_batch(params=[{}], **TIGHT)
    assert len(own) == 3
    model.set_param("E", 0.0)
    model.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="crossing time"):
        sim.run_batch(params=[{}], **TIGHT)
    model.set_param("E", 0.25)
    model.set_param("k", 1.0)
    model.reset()
    _close(_columns(sim.run_batch(params=[{}], **TIGHT)[0]), _gated(0.25))


REPORTED_ONLY = (
    "species A, B; A = 10; B = 0; k = 1; P = 2.5;\n"
    "F := piecewise(1, time >= floor(P) + 0.5, 0);\nR1: A -> B; k*A\n"
)


def test_a_condition_no_rate_law_reads_has_no_bearing_on_the_code(monkeypatch):
    """Control. ``F`` is reported and no reaction reads it. Its threshold,
    ``floor(P) + 0.5``, is one the pass cannot compensate, at every run and
    with nothing written, and the code is built by the rate laws alone: no run
    builds anything, and a batch row and every column at once run."""
    model = _model(REPORTED_ONLY)
    sim = _sim(model, ["k", "P"])
    assert sim.has_analytic_sens_rhs
    want = [-10.0 * np.exp(-1.0), 0.0]
    builds = _count_builds(monkeypatch)
    for _ in range(3):
        _close(_columns(sim.run(**TIGHT)), want)
        model.reset()
    rows = sim.run_batch(params=[{}, {"k": 2.0}], **TIGHT)
    _close(_columns(rows[0]), want)
    _close(_columns(rows[1]), [-10.0 * np.exp(-2.0), 0.0])
    _close(_columns(sim.compute_all_sensitivities(params=["k", "P"], **TIGHT)), want)
    assert builds == [0]


def test_rate_laws_that_cannot_be_read_do_not_let_the_old_code_through(monkeypatch):
    """The pass's finding is confirmed against the rate laws. Where those
    cannot be read for a crossing, the condition is taken to be read otherwise:
    the run is refused, and not let through on the code from before."""
    from bngsim import _switch_sensitivity

    model = _model(GATED)
    sim = _sim(model, ["k", "E"])
    model.set_param("E", 0.0)
    model.reset()

    def unread(*args, **kwargs):
        raise ValueError("not read")

    monkeypatch.setattr(_switch_sensitivity, "model_uncompensated_crossing_reason", unread)
    with pytest.raises(bngsim.SensitivityUnsupportedError):
        sim.run(**TIGHT)


def test_a_chunk_that_finds_the_condition_itself_is_a_refusal_by_name(monkeypatch):
    """``compute_all_sensitivities`` asks the conditions of its own model before
    it cuts a chunk. Were that pass to miss one, the chunk's finding is a
    ``SensitivityUnsupportedError`` that names the condition and the issue, and
    not a failed chunk to be retried column by column."""
    model = _model(GATED)
    sim = _sim(model, ["k", "E"])
    model.set_param("E", 0.0)
    model.reset()
    monkeypatch.setattr(
        bngsim.Simulator, "_sync_codegen_with_the_conditions", lambda *a, **k: None
    )
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"time\(\)>=.*\(issue #708\)"):
        sim.compute_all_sensitivities(params=["k", "E"], **TIGHT)


def test_a_counter_that_becomes_a_clock_runs_on_the_code_it_has(monkeypatch):
    """Control. ``-> C`` at ``one``, and a law that switches on ``C > 3.4``.
    At ``one = 2`` the species is a state the solver roots on, and at 1 it is
    a clock with a switch time. Either is compensated, by the code built for
    the other too, and nothing is built again: dY/dkb = 6 - 3.4/one."""
    text = (
        "species C, Y; C = 0; Y = 0; one = 2; kb = 1;\n"
        "J0: -> C; one\nJ1: -> Y; piecewise(kb, C > 3.4, 0)\n"
    )
    model = _model(text)
    sim = _sim(model, ["kb"])
    run = {"t_span": (0.0, 6.0), "n_points": 3, "rtol": 1e-9, "atol": 1e-12}
    np.testing.assert_allclose(_columns(sim.run(**run), "Y"), [6.0 - 1.7], rtol=1e-6)
    calls = _count_builds(monkeypatch)
    model.reset()
    model.set_param("one", 1.0)
    model.reset()
    np.testing.assert_allclose(_columns(sim.run(**run), "Y"), [2.6], rtol=1e-6)
    rows = sim.run_batch(params=[{"one": 2.0}, {"one": 1.0}], **run)
    np.testing.assert_allclose(_columns(rows[0], "Y"), [6.0 - 1.7], rtol=1e-6)
    np.testing.assert_allclose(_columns(rows[1], "Y"), [2.6], rtol=1e-6)
    assert calls[0] == 0


def test_a_condition_the_solver_roots_builds_nothing(monkeypatch):
    """Control. ``C >= sqrt(E)`` with ``C`` a counter species: at ``E = 0`` the
    switch-time pass cannot compensate it, and the solver locates it as a root
    of the state all the same, as it does in a model loaded there. The code is
    what a build at those values gives, and is kept. The rate laws are not
    scanned over it either: a rooted condition is not one to confirm."""
    from bngsim import _switch_sensitivity

    text = (
        "species C, Y; C = 0; Y = 0; one = 1; kb = 1; E = 0.25;\n"
        "J0: -> C; one\nJ1: -> Y; piecewise(kb, C >= sqrt(E), 0)\n"
    )
    model = _model(text)
    sim = _sim(model, ["kb", "E"])
    run = {"t_span": (0.0, 2.0), "n_points": 3, "rtol": 1e-9, "atol": 1e-12}
    np.testing.assert_allclose(_columns(sim.run(**run), "Y"), [1.5, -1.0], rtol=1e-6)
    calls = _count_builds(monkeypatch)
    scans = [0]
    real = _switch_sensitivity.model_uncompensated_crossing_reason

    def scanned(*args, **kwargs):
        scans[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(_switch_sensitivity, "model_uncompensated_crossing_reason", scanned)
    for value in (0.0, 0.25):
        model.reset()
        model.set_param("E", value)
        model.reset()
        sim.run(**run)
    assert calls[0] == 0
    assert scans[0] == 0
    assert sim.has_analytic_sens_rhs
    model.reset()
    np.testing.assert_allclose(_columns(sim.run(**run), "Y"), [1.5, -1.0], rtol=1e-6)


def test_a_batch_row_that_changes_how_a_condition_is_read_is_refused():
    """A row at ``E = 0.36`` reads the condition as the batch's build did and
    runs. One at ``E = 0`` does not, and returned dA/dE = 0."""
    sim = _sim(_model(GATED), ["k", "E"])
    rows = sim.run_batch(params=[{"E": 0.36}], **TIGHT)
    _close(_columns(rows[0]), _gated(0.36))
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#708") as refusal:
        sim.run_batch(params=[{"E": 0.36}, {"E": 0.0}], **TIGHT)
    assert "row 1" in str(refusal.value)
    assert "rate-law condition" in str(refusal.value)


def test_a_batch_on_the_quotient_asks_each_row_what_a_run_is_asked():
    """Control. ``-> C`` at ``one``: at 2, ``C`` is a species like another, and
    at 1 it is a clock, so the row reads ``C > 3.4`` otherwise than the batch's
    build did. That build has no analytic right-hand side (a ``max`` in another
    law), each row is asked what a run on the difference quotient is asked
    (issue #938), and this one passes: nothing requested moves the clock.
    dY/dkb = 6 - 3.4."""
    text = (
        "species C, Y, Z; C = 0; Y = 0; Z = 0; one = 2; kb = 1; kc = 0.1;\n"
        "J0: -> C; one\nJ1: -> Y; piecewise(kb, C > 3.4, 0)\nJ2: -> Z; kc*max(Y, 0.5)\n"
    )
    sim = _sim(_model(text), ["kb"])
    assert not sim.has_analytic_sens_rhs
    rows = sim.run_batch(
        params=[{"one": 1.0}], t_span=(0.0, 6.0), n_points=3, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(_columns(rows[0], "Y"), [2.6], rtol=1e-6)


def test_a_write_to_a_parameter_no_condition_reads_builds_nothing(monkeypatch):
    """Control. The gate is asked at each run of a model with a condition in a
    rate law, and a rate constant does not move its answer."""
    model = _model(GATED)
    sim = _sim(model, ["k", "E"])
    calls = _count_builds(monkeypatch)
    for k in (0.5, 2.0):
        model.reset()
        model.set_param("k", k)
        a = 10.0 * np.exp(-0.5 * k)
        _close(_columns(sim.run(**TIGHT)), [-0.5 * a, k * a])
    assert calls[0] == 0


# ── A batch row, by the columns that are asked for ───────────────────────────


def test_a_batch_row_that_pins_a_parameter_is_refused_by_the_columns_it_reaches():
    """The message names the columns that reach the model through the pinned
    parameter."""
    sim = _sim(_model(CHAIN), ["Q", "R0", "Rt"])
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#708") as refusal:
        sim.run_batch(params=[{"Rt": 5.0}], t_span=(0.0, 2.0), n_points=3)
    assert "['Q', 'R0']" in str(refusal.value)


def test_a_batch_row_that_pins_a_parameter_no_requested_column_reads_runs():
    """Control. The pinned parameter's own column, and the column of a
    parameter that does not reach the model through it, are the same code
    either way. ``-> A`` at ``k2``, ``A ->`` at ``kd*A``:
    A = (k2/kd)*(1 - exp(-kd*t))."""
    model = _model(STEADY)
    rows = _sim(model, ["kd", "k2"]).run_batch(params=[{"k2": 5.0}], **RUN)
    kd, k2 = 0.5, 5.0
    decay = np.exp(-kd)
    _close(_columns(rows[0]), [k2 * (decay / kd - (1.0 - decay) / kd**2), (1.0 - decay) / kd])
    rows = _sim(_model(CHAIN), ["Rt"]).run_batch(
        params=[{"Rt": 5.0}], t_span=(0.0, 2.0), n_points=3
    )
    _close(np.asarray(rows[0].sensitivities)[-1, 0], [2.0])


# ── What is recorded beside an artifact, and what that saves ─────────────────


def test_every_column_at_once_on_a_sensitivity_simulator_builds_nothing(monkeypatch):
    """Control. Its artifact was built for a sensitivity run and the model is
    as it was."""
    model = _model()
    sim = _sim(model)
    calls = _count_builds(monkeypatch)
    for _ in range(2):
        model.reset()
        _close(_columns(sim.compute_all_sensitivities(params=P, **RUN)), ATTACHED)
    assert calls[0] == 0


def test_a_simulator_on_a_clone_takes_the_artifact_the_model_carries(monkeypatch):
    """Control. A clone carries the artifact with what it was built for, and a
    sensitivity Simulator on it builds nothing, at construction or when every
    column is asked for at once."""
    model = _model()
    first = _sim(model)
    calls = _count_builds(monkeypatch)
    clone = model.clone()
    sim = _sim(clone)
    assert (sim._codegen_so_path, sim._codegen_c_source) == (
        first._codegen_so_path,
        first._codegen_c_source,
    )
    _close(_columns(sim.compute_all_sensitivities(params=P, **RUN)), ATTACHED)
    assert calls[0] == 0


def test_a_sensitivity_simulator_does_not_take_an_artifact_built_without_being_asked_for(
    monkeypatch,
):
    """Control. A model at the size threshold is compiled for a plain run, and
    that artifact has no sensitivity code. A sensitivity Simulator on the model
    builds its own."""
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
    model = _model()
    plain = bngsim.Simulator(model, method="ode")
    assert plain._codegen_so_path or plain._codegen_c_source
    sim = _sim(model)
    assert sim.has_analytic_sens_rhs
    _close(_columns(sim.run(**RUN)), ATTACHED)
