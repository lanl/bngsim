"""Issue #1366: ``bngsim.run_protocol`` runs a BNGL file's actions as BNG2.pl does.

Each test runs actions on a hand-written network of ``A() -> B()`` at rate ``k``
(``A0`` = 100 of ``A`` to start), where ``A(t) = A(0) exp(-k t)``, so the value a
step should leave is known in closed form and the tests need no BioNetGen. What
BNG2.pl itself does is in ``Perl2/BNGModel.pm`` and ``BNGAction.pm`` (2.9.3);
each test names the rule it holds.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import ConversionError, ConversionWarning
from bngsim.convert import parse_bngl_protocol

NET = """\
begin parameters
    1 k   1
    2 A0  100
    3 kp  0
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 k
end reactions
begin groups
    1 Atot 1
    2 Btot 2
end groups
"""

RTOL = 1e-6  # the runs use run_network's 1e-8 tolerances


def _model(tmp_path: Path, net: str = NET) -> bngsim.Model:
    path = tmp_path / "m.net"
    path.write_text(net)
    return bngsim.Model.from_net(str(path))


def _spec(*actions: str, strict: bool = True):
    text = "begin model\nend model\n" + "\n".join(actions) + "\n"
    return parse_bngl_protocol(text, strict=strict)


def _run(tmp_path: Path, *actions: str, net: str = NET, **kwargs):
    gen = ("generate_network({overwrite=>1})",)
    return bngsim.run_protocol(_model(tmp_path, net), _spec(*gen, *actions), **kwargs)


def _a(result) -> np.ndarray:
    return np.asarray(result.observables["Atot"])


def _sim(t_end: float, **extra: object) -> str:
    args = "".join(f",{k}=>{v}" for k, v in extra.items())
    return f'simulate({{method=>"ode",t_end=>{t_end},n_steps=>10{args}}})'


# --------------------------------------------------------------------------- #
# runs and the state between them
# --------------------------------------------------------------------------- #
def test_each_run_starts_from_the_state_the_last_left(tmp_path):
    """BNG2.pl reads each run's end state back as the concentrations, and a run
    without continue starts at t_start, 0."""
    out = _run(tmp_path, _sim(1), _sim(1))
    first, second = out[0].result, out[1].result
    assert second.time[0] == 0.0
    assert _a(second)[0] == pytest.approx(100 * math.exp(-1), rel=RTOL)
    assert _a(second)[-1] == pytest.approx(100 * math.exp(-2), rel=RTOL)
    assert _a(first)[-1] == pytest.approx(_a(second)[0], rel=1e-12)


def test_continue_starts_at_the_time_the_last_run_left(tmp_path):
    out = _run(tmp_path, _sim(1), _sim(3, **{"continue": 1}))
    assert (out[1].result.time[0], out[1].result.time[-1]) == (1.0, 3.0)
    assert _a(out[1].result)[-1] == pytest.approx(100 * math.exp(-3), rel=RTOL)


def test_the_model_given_is_left_as_it_was(tmp_path):
    model = _model(tmp_path)
    out = bngsim.run_protocol(model, _spec("generate_network()", 'setParameter("k",3)', _sim(2)))
    assert model.get_param("k") == 1.0 and model.get_concentration("A()") == 100.0
    assert out.model.get_param("k") == 3.0
    assert out.model.get_concentration("A()") == pytest.approx(100 * math.exp(-6), rel=RTOL)


# --------------------------------------------------------------------------- #
# setParameter, setConcentration, addConcentration
# --------------------------------------------------------------------------- #
def test_set_parameter_and_set_concentration_change_the_next_run(tmp_path):
    out = _run(
        tmp_path,
        'setParameter("kp",2)',
        'setParameter("k","kp")',
        'setConcentration("A()",50)',
        _sim(1),
    )
    assert _a(out[0].result)[0] == 50.0
    assert _a(out[0].result)[-1] == pytest.approx(50 * math.exp(-2), rel=RTOL)


def test_add_concentration_adds_to_what_the_last_run_left(tmp_path):
    out = _run(tmp_path, _sim(1), 'addConcentration("A()",10)', _sim(1))
    assert _a(out[1].result)[0] == pytest.approx(100 * math.exp(-1) + 10, rel=RTOL)


def test_add_concentration_adds_to_an_amount_still_its_expression(tmp_path):
    """BNG2.pl evaluates the species' expression first (BNGModel.pm,
    addConcentration: ``$plist->evaluate($spec->Concentration)``): A0 is 10
    here, so the amount is 15, not the 105 the network's last number makes."""
    out = _run(tmp_path, 'setParameter("A0",10)', 'addConcentration("A()",5)', _sim(1))
    assert _a(out[0].result)[0] == 15.0


def test_a_seed_amount_follows_its_parameter_until_a_run(tmp_path):
    """The network writes ``A() A0``, an expression BNG2.pl writes again into
    each run's network until a run makes it a number."""
    out = _run(tmp_path, 'setParameter("A0",10)', _sim(1), 'setParameter("A0",5)', _sim(1))
    assert _a(out[0].result)[0] == 10.0
    assert _a(out[1].result)[0] == pytest.approx(10 * math.exp(-1), rel=RTOL)


def test_an_amount_set_to_an_expression_follows_it_until_a_run(tmp_path):
    out = _run(tmp_path, 'setConcentration("A()","2*kp")', 'setParameter("kp",7)', _sim(1))
    assert _a(out[0].result)[0] == 14.0


def test_an_amount_set_to_a_number_does_not_follow_a_parameter(tmp_path):
    """Control: ``setConcentration("A()",100)`` is the number 100, though it equals
    ``A0``. bngsim's set_param would move a species still at its initial amount;
    BNG2.pl moves only one still at the expression."""
    out = _run(tmp_path, 'setConcentration("A()",100)', 'setParameter("A0",10)', _sim(1))
    assert _a(out[0].result)[0] == 100.0


def test_a_species_the_network_does_not_have_is_refused(tmp_path):
    with pytest.raises(ConversionError, match=r"no species 'C\(\)'"):
        _run(tmp_path, 'setConcentration("C()",1)', _sim(1))


# --------------------------------------------------------------------------- #
# save and reset
# --------------------------------------------------------------------------- #
def test_a_reset_restores_the_state_saved_under_its_label(tmp_path):
    out = _run(
        tmp_path,
        _sim(1),
        'saveConcentrations("one")',
        _sim(1),
        "saveConcentrations()",
        'resetConcentrations("one")',
        _sim(1),
    )
    assert _a(out[2].result)[0] == pytest.approx(100 * math.exp(-1), rel=RTOL)


def test_a_reset_with_nothing_saved_is_the_model_as_declared(tmp_path):
    """``A() A0`` again, at the A0 of the moment (BNGModel.pm: the species' own
    Concentration, an expression)."""
    out = _run(tmp_path, _sim(1), 'setParameter("A0",40)', "resetConcentrations()", _sim(1))
    assert _a(out[1].result)[0] == 40.0


def test_a_reset_to_a_label_never_saved_is_refused(tmp_path):
    with pytest.raises(ConversionError, match=r"nothing saved under 'x'"):
        _run(tmp_path, 'resetConcentrations("x")', _sim(1))


def test_reset_parameters_restores_the_saved_parameters(tmp_path):
    out = _run(tmp_path, "saveParameters()", 'setParameter("k",5)', "resetParameters()", _sim(1))
    assert _a(out[0].result)[-1] == pytest.approx(100 * math.exp(-1), rel=RTOL)


def test_reset_parameters_with_nothing_saved_is_refused(tmp_path):
    """BNG2.pl's resetParameters dies on an empty cache, even for the default label."""
    with pytest.raises(ConversionError, match=r"resetParameters\(\): nothing saved"):
        _run(tmp_path, 'setParameter("k",5)', "resetParameters()", _sim(1))


def test_reset_concentrations_leaves_parameters_alone(tmp_path):
    out = _run(
        tmp_path, "saveConcentrations()", 'setParameter("k",2)', "resetConcentrations()", _sim(1)
    )
    assert _a(out[0].result)[-1] == pytest.approx(100 * math.exp(-2), rel=RTOL)


# --------------------------------------------------------------------------- #
# scans and bifurcations
# --------------------------------------------------------------------------- #
_SCAN = (
    'parameter_scan({{method=>"ode",parameter=>"k",par_scan_vals=>[1,3,2],t_end=>1,n_steps=>5{}}})'
)


def test_a_scan_runs_each_value_from_the_same_state(tmp_path):
    out = _run(tmp_path, _SCAN.format(""), _sim(1))
    scan = out[0]
    assert scan.scan_values == (1.0, 3.0, 2.0)
    ends = [_a(r)[-1] for r in scan.scan_results]
    assert ends == pytest.approx([100 * math.exp(-k) for k in (1, 3, 2)], rel=RTOL)
    assert [r.custom_attrs["scan_value"] for r in scan.scan_results] == [1.0, 3.0, 2.0]
    # Then the state is reset, and k keeps its last value, 2 (BNGAction.pm).
    after = out[1].result
    assert _a(after)[0] == 100.0
    assert _a(after)[-1] == pytest.approx(100 * math.exp(-2), rel=RTOL)


def test_a_scan_without_reset_continues_point_to_point(tmp_path):
    out = _run(tmp_path, _SCAN.format(",reset_conc=>0"))
    ends = [_a(r)[-1] for r in out[0].scan_results]
    assert ends == pytest.approx(
        [100 * math.exp(-1), 100 * math.exp(-4), 100 * math.exp(-6)], rel=RTOL
    )


def test_a_scan_range_is_bng2s_arithmetic(tmp_path):
    out = _run(
        tmp_path,
        'parameter_scan({method=>"ode",parameter=>"k",par_min=>0.1,par_max=>10,n_scan_pts=>3,'
        "log_scale=>1,t_end=>1})",
    )
    assert out[0].scan_values == pytest.approx((0.1, 1.0, 10.0), rel=1e-12)


def test_a_bifurcation_scans_up_then_back_down_from_where_it_got(tmp_path):
    out = _run(
        tmp_path,
        'bifurcate({method=>"ode",parameter=>"k",par_min=>1,par_max=>2,n_scan_pts=>2,t_end=>1})',
    )
    run = out[0]
    assert (run.scan_values, run.backward_values) == ((1.0, 2.0), (2.0, 1.0))
    a = 100 * math.exp(-1) * math.exp(-2)
    assert _a(run.backward_results[0])[0] == pytest.approx(a, rel=RTOL)
    assert _a(run.backward_results[-1])[-1] == pytest.approx(a * math.exp(-3), rel=RTOL)


# --------------------------------------------------------------------------- #
# run arguments
# --------------------------------------------------------------------------- #
def test_uneven_sample_times_are_the_output_times(tmp_path):
    out = _run(tmp_path, 'simulate({method=>"ode",sample_times=>[0.5,1,4]})')
    assert list(out[0].result.time) == [0.0, 0.5, 1.0, 4.0]


def test_a_steady_state_run_stops_there_and_the_next_continues_from_it(tmp_path):
    """The spec cannot say where the second run starts (it is lossy there, with
    no approximation of the run); the replay knows, and says nothing."""
    with pytest.warns(ConversionWarning, match=r"continue=>1 starts at the model time"):
        spec = _spec(
            "generate_network()",
            _sim(1e4, steady_state=1),
            _sim(2e4, **{"continue": 1}),
            strict=False,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        out = bngsim.run_protocol(_model(tmp_path), spec)
    stop = out[0].result.time[-1]
    assert stop < 1e4
    assert out[1].result.time[0] == stop


def test_stochastic_runs_are_seeded_from_the_seed_given(tmp_path):
    actions = ('simulate({method=>"ssa",t_end=>1,n_steps=>5})',) * 2
    one = _run(tmp_path, *actions, seed=7)
    two = _run(tmp_path, *actions, seed=7)
    for a, b in zip(one, two, strict=True):
        assert np.array_equal(_a(a.result), _a(b.result))
    assert not np.array_equal(_a(one[0].result), _a(one[1].result))


def test_poplevel_runs_psa(tmp_path):
    out = _run(tmp_path, 'simulate({method=>"ssa",t_end=>1,n_steps=>5,poplevel=>10})', seed=1)
    assert out[0].method == "psa"
    assert np.all(np.isfinite(_a(out[0].result)))


# --------------------------------------------------------------------------- #
# where the replay starts, and what it refuses
# --------------------------------------------------------------------------- #
def test_actions_before_generation_are_not_applied_again(tmp_path):
    """The network from_bngl loads already holds them: here it holds the 110 an
    ``addConcentration`` before generation left, and adding again would be 120."""
    model = _model(tmp_path, NET.replace("1 A() A0", "1 A() 110"))
    spec = _spec('addConcentration("A()",10)', "generate_network()", _sim(1))
    assert spec.generation_step == 1
    out = bngsim.run_protocol(model, spec)
    assert _a(out[0].result)[0] == 110.0


def test_a_reset_to_a_seed_changed_before_generation_is_refused(tmp_path):
    """BNG2.pl's reset with nothing saved goes back to the seed as declared, which
    the loaded network no longer holds."""
    model = _model(tmp_path, NET.replace("1 A() A0", "1 A() 110"))
    spec = _spec('addConcentration("A()",10)', "generate_network()", "resetConcentrations()")
    with pytest.raises(ConversionError, match=r"an action before generation changed them"):
        bngsim.run_protocol(model, spec)


@pytest.mark.parametrize(
    ("action", "why"),
    [
        ('simulate({method=>"nf",t_end=>5,n_steps=>5})', "network-free"),
        ('simulate({method=>"pla",t_end=>1,n_steps=>5})', "PLA"),
        ('simulate({method=>"ode",t_end=>1,stop_if=>"Atot<1"})', "stop_if"),
    ],
    ids=["nfsim", "pla", "stop_if"],
)
def test_a_run_bngsim_cannot_do_is_refused_before_anything_runs(tmp_path, action, why):
    model = _model(tmp_path)
    spec = _spec("generate_network()", 'setConcentration("A()",1)', _sim(1), action)
    with pytest.raises(ConversionError, match=why):
        bngsim.run_protocol(model, spec)


def test_an_action_the_protocol_cannot_hold_is_refused_unless_allowed(tmp_path):
    with pytest.warns(ConversionWarning):
        spec = parse_bngl_protocol(
            'begin model\nend model\ngenerate_network()\nsetVolume("c",2)\n' + _sim(1) + "\n",
            strict=False,
        )
    with pytest.raises(ConversionError, match=r"setVolume changes results"):
        bngsim.run_protocol(_model(tmp_path), spec)
    with pytest.warns(ConversionWarning, match=r"as nearly as it holds it"):
        out = bngsim.run_protocol(_model(tmp_path), spec, allow_lossy=True)
    assert len(out) == 1


def test_runs_are_found_by_suffix(tmp_path):
    out = _run(tmp_path, _sim(1, suffix='"eq"'), _sim(1, suffix='"dose"'))
    assert out["dose"] is out[1]
    assert out.labels == ["eq", "dose"]
    with pytest.raises(KeyError, match=r"no run has the suffix 'x'"):
        out["x"]


def test_an_action_before_generation_is_not_refused(tmp_path):
    """A setOption before generate_network is in the network BNG2.pl generated;
    after it, it is an action the replay cannot apply."""
    opt = 'setOption("NumberPerQuantityUnit",6.0221e23)'
    with pytest.warns(ConversionWarning):
        before = _spec(opt, "generate_network()", _sim(1), strict=False)
        after = _spec("generate_network()", opt, _sim(1), strict=False)
    assert len(bngsim.run_protocol(_model(tmp_path), before)) == 1
    with pytest.raises(ConversionError, match=r"setOption changes results"):
        bngsim.run_protocol(_model(tmp_path), after)


def test_a_file_writer_is_dropped_cleanly(tmp_path):
    """BNG2.pl's writers (writeLatex, writeSSC, ...) only write files."""
    spec = _spec("generate_network()", "writeLatex()", "writeSSC()", _sim(1))
    assert spec.dropped == ("generate_network", "writeLatex", "writeSSC")
    assert len(bngsim.run_protocol(_model(tmp_path), spec)) == 1


def test_reset_parameters_leaves_function_values_alone(tmp_path):
    """BNG2.pl saves constants only (copyConstant). bngsim keeps a function's
    value in a parameter slot, which a run changes and which cannot be set."""
    net = NET.replace(
        "end groups\n", "end groups\nbegin functions\n    1 frac() Atot/A0\nend functions\n"
    )
    out = _run(tmp_path, "saveParameters()", _sim(1), "resetParameters()", _sim(1), net=net)
    assert len(out) == 2


def test_a_reset_before_generation_that_changes_nothing_is_no_change(tmp_path):
    """``resetConcentrations()`` with nothing saved, before generation, is the seed
    as declared (prion2_YTLedits.bngl opens with one); a reset after it can still
    go back there."""
    spec = _spec(
        "resetConcentrations()", "generate_network()", _sim(1), "resetConcentrations()", _sim(1)
    )
    out = bngsim.run_protocol(_model(tmp_path), spec)
    assert _a(out[1].result)[0] == 100.0


def test_a_save_before_generation_holds_the_loaded_state_until_changed(tmp_path):
    model = _model(tmp_path, NET.replace("1 A() A0", "1 A() 110"))
    held = _spec(
        'setConcentration("A()",110)',
        'saveConcentrations("s")',
        "generate_network()",
        _sim(1),
        'resetConcentrations("s")',
        _sim(1),
    )
    out = bngsim.run_protocol(model, held)
    assert _a(out[1].result)[0] == 110.0
    changed = _spec(
        'saveConcentrations("s")',
        'setConcentration("A()",110)',
        "generate_network()",
        'resetConcentrations("s")',
    )
    with pytest.raises(ConversionError, match=r"saved under 's' before the network"):
        bngsim.run_protocol(model, changed)


# --------------------------------------------------------------------------- #
# the blind review's findings (issue #1366), each against BNG2.pl 2.9.3
# --------------------------------------------------------------------------- #
def test_add_concentration_to_an_expression_amount_adds_to_the_declared_one(tmp_path):
    """BNG2.pl adds to the species' declared amount, not to the expression a
    setConcentration left (ran: 'Add 5 ... for total of 105', and 5 for B)."""
    out = _run(
        tmp_path,
        'setParameter("kp",7)',
        'setConcentration("A()","2*kp")',
        'addConcentration("A()",5)',
        'setConcentration("B()","kp")',
        'addConcentration("B()",5)',
        _sim(1),
    )
    assert _a(out[0].result)[0] == 105.0
    assert out[0].result.observables["Btot"][0] == 5.0


def test_set_parameter_to_an_observable_is_refused(tmp_path):
    """BNG2.pl makes kp a function of Atot, read through every later run (ran:
    A(1) = 50 for A' = -kp A, kp = Atot/100); bngsim cannot make a parameter one."""
    with pytest.raises(ConversionError, match=r"reads Atot, which makes the parameter"):
        _run(tmp_path, _sim(1), 'setParameter("k","Atot/100")', _sim(1))


def test_an_amount_still_an_expression_counts_as_0_in_an_observable(tmp_path):
    """Before any run, A is the string A0 in BNG2.pl's concentrations, which its
    Observable::evaluate reads as 0 (ran: 'Set concentration ... to value 0')."""
    out = _run(tmp_path, 'setConcentration("B()","Atot")', _sim(1))
    assert out[0].result.observables["Btot"][0] == 0.0


def test_an_observable_in_a_value_is_read_from_the_amounts_now(tmp_path):
    """BNG2.pl's Observable::evaluate sums the amounts of the moment; bngsim's
    observable slot still holds the last run's value."""
    out = _run(
        tmp_path, _sim(1), 'setConcentration("A()",0)', 'setConcentration("B()","Atot")', _sim(1)
    )
    assert out[1].result.observables["Btot"][0] == 0.0


def test_sample_times_after_an_early_stop_are_the_samples(tmp_path):
    """The parse cannot know where a steady-state run stops, so it cannot tell
    whether the samples after it are even; the replay outputs at them."""
    with pytest.warns(ConversionWarning):
        spec = _spec(
            "generate_network()",
            _sim(1e5, steady_state=1),
            'simulate({method=>"ode",continue=>1,sample_times=>[1e5,2e5,3e5]})',
            strict=False,
        )
    out = bngsim.run_protocol(_model(tmp_path), spec)
    stop = out[0].result.time[-1]
    assert list(out[1].result.time) == [stop, 1e5, 2e5, 3e5]


def test_a_run_of_no_time_writes_one_row_and_makes_amounts_numbers(tmp_path):
    """BNG2.pl warns, runs run_network for the one row at t_start, and reads the
    amounts back, so A no longer follows A0."""
    with pytest.warns(ConversionWarning, match=r"writes the one row at t_start"):
        out = _run(
            tmp_path,
            'simulate({method=>"ode",t_start=>5,t_end=>5})',
            'setParameter("A0",10)',
            _sim(1),
        )
    assert list(out[0].result.time) == [5.0] and _a(out[0].result)[0] == 100.0
    assert _a(out[1].result)[0] == 100.0


def test_a_continuation_from_the_wrong_time_is_refused(tmp_path):
    """BNG2.pl: 't_start must equal current model time for continuation'. After a
    run that can stop early, only the replay knows that time."""
    spec = _spec(
        "generate_network()",
        _sim(1e5, steady_state=1),
        'simulate({method=>"ode",continue=>1,t_start=>1e5,t_end=>2e5})',
    )
    with pytest.raises(ConversionError, match=r"continue=>1 with t_start=100000"):
        bngsim.run_protocol(_model(tmp_path), spec)


@pytest.mark.parametrize(
    ("action", "why"),
    [
        (
            'parameter_scan({method=>"ode",parameter=>"k",par_min=>1,par_max=>2,'
            "n_scan_pts=>1,t_end=>1})",
            "more than one point",
        ),
        ('simulate({method=>"ode",t_end=>1,n_steps=>0})', "n_steps must be 1 or more"),
    ],
    ids=["one point over a range", "no steps"],
)
def test_what_bng2_refuses_is_refused(tmp_path, action, why):
    with pytest.raises(ConversionError, match=why):
        _run(tmp_path, action)


CNET = """\
begin parameters
    1 k 1
end parameters
begin compartments
    1 c 3 1
end compartments
begin species
    1 @c::A(b,x) 10
end species
begin reactions
    1 1 0 k
end reactions
begin groups
    1 Atot 1
end groups
"""


@pytest.mark.parametrize("pattern", ["@c:A(x,b)", "A(b,x)@c", "@c::A(b,x)"])
def test_a_compartment_species_is_found_however_it_is_written(tmp_path, pattern):
    """BNGL actions write @c:A() or A()@c; the network writes @c::A()."""
    out = _run(tmp_path, f'setConcentration("{pattern}",50)', _sim(1), net=CNET)
    assert _a(out[0].result)[0] == 50.0


def test_a_complex_with_a_compartment_on_each_molecule_is_found():
    """BNG2.pl reads A(b!1)@c.B(a!1)@c as @c::A(b!1).B(a!1) (ran)."""
    from bngsim._bngl_loader import _network_spelling

    assert _network_spelling("A(b!1)@c.B(a!1)@c") == "@c::A(b!1).B(a!1)"
    assert _network_spelling("A(b!1)@c.B(a!1)@m") == "A(b!1)@c.B(a!1)@m"
