"""Issue #1364: parse_bngl_protocol reads simulate/parameter_scan as BNG2.pl does.

``BNGAction.pm`` (BNG2 2.9.3) defines the arguments: ``continue=>1`` starts at
the model time, where an earlier run ended; ``n_output_steps`` is ``n_steps``,
both is an error, and neither is one step; ``sample_times`` sets the grid where
neither is given, ending at its last time; a scan reads ``par_scan_vals`` where
``par_min``/``par_max`` are not given. The parser had none of this: (0, 20) for
a continued (10, 20), 101 points for 11 and for 2, a horizon of 0, and a scan of
one run, and ``strict=True`` raised nothing.

What the IR's uniform ``(t_span, n_points)`` and ``(min, max, points, log)``
hold, it now holds exactly; what they cannot (an uneven grid, uneven scan
values) is lossy, and ``strict=True`` refuses it.

The sample grid is what ``run_network`` writes: ``t_start`` first, then the
samples strictly between it and the end, which is a given ``t_end`` (BNG2.pl
appends it) or else the last sample. What BNG2.pl refuses (fewer than three
sample times, a continuation that does not start at the model time) is refused
here too.
"""

from __future__ import annotations

import re

import pytest
from bngsim import ConversionError, ConversionWarning
from bngsim.convert import parse_bngl_protocol, write_bngl_protocol, write_sedml_protocol


def _parse(*actions: str, strict: bool = True):
    return parse_bngl_protocol(
        "begin actions\n" + "\n".join(actions) + "\nend actions\n", strict=strict
    )


def _only(*actions: str):
    spec = _parse(*actions)
    assert spec.lossy == ()
    return spec.steps


def test_a_continued_run_starts_where_the_last_one_ended():
    first, second = _only(
        'simulate({method=>"ode",t_end=>10,n_steps=>10})',
        'simulate({method=>"ode",continue=>1,t_end=>20,n_steps=>10})',
    )
    assert first.t_span == (0.0, 10.0)
    assert second.t_span == (10.0, 20.0)


def test_a_continued_run_after_a_scan_starts_at_the_scans_end():
    """Each point of a scan is a simulate, which leaves the model time at its
    t_end."""
    _, cont = _only(
        'parameter_scan({parameter=>"k",par_min=>1,par_max=>2,n_scan_pts=>2,t_end=>5})',
        'simulate({method=>"ode",continue=>1,t_end=>8,n_steps=>3})',
    )
    assert cont.t_span == (5.0, 8.0)


def test_a_continued_run_with_its_t_start_keeps_it():
    _, cont = _only(
        'simulate({method=>"ode",t_end=>10,n_steps=>10})',
        'simulate({method=>"ode",continue=>1,t_start=>10,t_end=>30,n_steps=>4})',
    )
    assert cont.t_span == (10.0, 30.0)


def test_a_continuation_from_an_unknown_time_is_lossy():
    with pytest.raises(ConversionError, match=r"continue=>1 starts at the model time"):
        _parse(
            'simulate({method=>"ode",t_end=>tf,n_steps=>10})',
            'simulate({method=>"ode",continue=>1,t_end=>20,n_steps=>10})',
        )


def test_n_output_steps_is_n_steps():
    (exp,) = _only('simulate({method=>"ode",t_end=>10,n_output_steps=>10})')
    assert exp.n_points == 11
    assert "n_output_steps" not in exp.extra


def test_both_n_steps_and_n_output_steps_is_refused():
    with pytest.raises(ConversionError, match=r"'n_steps' and 'n_output_steps'"):
        _parse('simulate({method=>"ode",t_end=>10,n_steps=>5,n_output_steps=>5})')


def test_no_n_steps_is_one_step():
    (exp,) = _only('simulate({method=>"ode",t_end=>10})')
    assert exp.n_points == 2


def test_an_n_steps_the_parse_cannot_read_keeps_its_fallback():
    """Control: a parameter reference is not BNG2's default; the 100 the parse
    has always taken for it is kept."""
    (exp,) = _only('simulate({method=>"ode",t_end=>10,n_steps=>nsteps})')
    assert exp.n_points == 101


def test_an_even_sample_grid_from_t_start_is_the_time_course():
    (exp,) = _only('simulate({method=>"ode",sample_times=>[0,2,4,6,8,10]})')
    assert exp.t_span == (0.0, 10.0)
    assert exp.n_points == 6
    assert "sample_times" not in exp.extra


def test_n_steps_takes_precedence_over_sample_times():
    (exp,) = _only('simulate({method=>"ode",t_end=>10,n_steps=>5,sample_times=>[0,1,5,10]})')
    assert (exp.t_span, exp.n_points) == ((0.0, 10.0), 6)


def test_an_uneven_sample_grid_is_lossy():
    with pytest.raises(
        ConversionError, match=r"sample_times give the uneven output times \[0.0, 1.0, 5.0, 10.0\]"
    ):
        _parse('simulate({method=>"ode",sample_times=>[0,1,5,10]})')
    with pytest.warns(ConversionWarning):
        spec = _parse('simulate({method=>"ode",sample_times=>[0,1,5,10]})', strict=False)
    assert spec.lossy == ("simulate",)
    assert spec.steps[0].t_span == (0.0, 10.0)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ("[1,2,3]", (1.0, 3.0, 3, False)),
        ("[0.1,1,10]", (0.1, 10.0, 3, True)),
        ("[3,1,2]", (1.0, 3.0, 3, False)),
    ],
    ids=["linear", "log", "unsorted"],
)
def test_par_scan_vals_are_the_scan(values, expected):
    (exp,) = _only(
        f'parameter_scan({{parameter=>"k",par_scan_vals=>{values},method=>"ode",t_end=>10,n_steps=>5}})'
    )
    assert (exp.scan_min, exp.scan_max, exp.scan_points, exp.scan_log) == pytest.approx(expected)
    assert "par_scan_vals" not in exp.extra


def test_par_min_and_par_max_take_precedence_over_par_scan_vals():
    (exp,) = _only(
        'parameter_scan({parameter=>"k",par_min=>1,par_max=>9,n_scan_pts=>5,'
        "par_scan_vals=>[1,2],t_end=>10})"
    )
    assert (exp.scan_min, exp.scan_max, exp.scan_points) == (1.0, 9.0, 5)


def test_uneven_par_scan_vals_are_lossy():
    with pytest.raises(
        ConversionError, match=r"par_scan_vals \[1.0, 2.0, 7.0\] is not evenly spaced"
    ):
        _parse('parameter_scan({parameter=>"k",par_scan_vals=>[1,2,7],t_end=>1})')


def test_the_sedml_export_carries_the_grid_and_the_scan():
    spec = _parse(
        'simulate({method=>"ode",t_end=>10,n_steps=>10})',
        'simulate({method=>"ode",continue=>1,t_end=>20,n_steps=>10})',
        'simulate({method=>"ode",sample_times=>[0,3,6,9,12]})',
        'parameter_scan({parameter=>"k",par_scan_vals=>[1,2,3],method=>"ode",t_end=>10,n_steps=>5})',
    )
    x = write_sedml_protocol(spec)
    grids = re.findall(
        r'outputStartTime="([^"]*)" outputEndTime="([^"]*)" numberOfPoints="([^"]*)"', x
    )
    assert grids == [("0", "10", "10"), ("10", "20", "10"), ("0", "12", "4"), ("0", "10", "5")]
    assert "repeatedTask" in x


def test_the_bngl_writer_round_trips_what_bng2_accepts():
    """Re-emitted, a continuation names the t_start it implies beside
    continue=>1 (BNG2.pl checks they agree), n_output_steps is one n_steps, and
    a scan is par_min/par_max/n_scan_pts; parsed again, the spec is the same.
    That BNG2.pl writes identical .gdat and .scan files from both is in the PR."""
    spec = _parse(
        'simulate({method=>"ode",t_end=>10,n_steps=>10})',
        'simulate({method=>"ode",continue=>1,t_end=>20,n_steps=>10})',
        'simulate({method=>"ode",t_end=>10,n_output_steps=>10})',
        'parameter_scan({parameter=>"k",par_scan_vals=>[0.1,1,10],method=>"ode",t_end=>10,n_steps=>5})',
    )
    text = write_bngl_protocol(spec)
    assert "t_start=>10,t_end=>20,n_steps=>10,continue=>1" in text
    assert "n_output_steps" not in text and "par_scan_vals" not in text
    assert parse_bngl_protocol(text, strict=True).steps == spec.steps


def test_sample_times_always_start_at_t_start():
    """``run_network`` writes a row at t_start whether or not it is a sample
    time, so samples from 1 make the even grid 0..10."""
    (exp,) = _only('simulate({method=>"ode",sample_times=>[1,2,3,4,5,6,7,8,9,10]})')
    assert (exp.t_span, exp.n_points) == ((0.0, 10.0), 11)


def test_a_given_t_end_ends_the_sample_grid():
    """BNG2.pl appends t_end to the sample times, so it ends the run even with
    samples past it, and the next continuation starts there."""
    first, cont = _only(
        'simulate({method=>"ode",t_end=>5,sample_times=>[0,1,2,3,4,5,6,7,8,9,10]})',
        'simulate({method=>"ode",continue=>1,t_end=>10,n_steps=>5})',
    )
    assert (first.t_span, first.n_points) == ((0.0, 5.0), 6)
    assert cont.t_span == (5.0, 10.0)


def test_a_continuation_drops_the_sample_times_before_it():
    _, cont = _only(
        'simulate({method=>"ode",t_end=>10,n_steps=>10})',
        'simulate({method=>"ode",continue=>1,t_end=>20,sample_times=>[0,5,10,15]})',
    )
    assert (cont.t_span, cont.n_points) == ((10.0, 20.0), 3)


def test_fewer_than_three_sample_times_is_refused():
    with pytest.raises(ConversionError, match=r"fewer than 3 times; BNG2.pl refuses it"):
        _parse('simulate({method=>"ode",sample_times=>[0,10]})', strict=False)


def test_sample_times_ending_at_an_unreadable_t_end_are_lossy():
    with pytest.raises(ConversionError, match=r"t_end that cannot be read without the model"):
        _parse('simulate({method=>"ode",t_end=>tf,sample_times=>[0,1,2,3]})')


def test_a_continuation_that_does_not_start_at_the_model_time_is_refused():
    with pytest.raises(ConversionError, match=r"refuses a continuation that does not start there"):
        _parse(
            'simulate({method=>"ode",t_end=>10,n_steps=>10})',
            'simulate({method=>"ode",continue=>1,t_start=>5,t_end=>20,n_steps=>10})',
            strict=False,
        )


def test_a_continuations_unreadable_t_start_is_the_model_time():
    _, cont = _only(
        'simulate({method=>"ode",t_end=>10,n_steps=>10})',
        'simulate({method=>"ode",continue=>1,t_start=>t0,t_end=>20,n_steps=>10})',
    )
    assert cont.t_span == (10.0, 20.0)


@pytest.mark.parametrize("stop", ["steady_state=>1", 'stop_if=>"A<1"', "max_sim_steps=>5"])
def test_a_run_that_can_stop_early_leaves_the_model_time_unknown(stop):
    with pytest.raises(ConversionError, match=r"continue=>1 starts at the model time"):
        _parse(
            f'simulate({{method=>"ode",t_end=>100,n_steps=>10,{stop}}})',
            'simulate({method=>"ode",continue=>1,t_end=>200,n_steps=>10})',
        )


@pytest.mark.parametrize(
    "second",
    ["simulate_nf({t_end=>20,n_steps=>10})", 'simulate({method=>"nf",t_end=>20,n_steps=>10})'],
)
def test_an_nf_run_starts_at_the_model_time(second):
    """``simulate_nf`` takes the model time as t_start with or without
    continue (``BNGAction.pm``)."""
    _, exp = _only('simulate({method=>"ode",t_end=>10,n_steps=>10})', second)
    assert exp.t_span == (10.0, 20.0)


def test_an_nf_run_leaves_the_model_time_undefined():
    """``simulate_nf``'s n_steps branch declares a second ``my $t_end``, so it
    sets the model time to undef, which the next start reads as 0: replicate
    nf runs (Dolan2015 has ten) each start at 0, as does a continuation. That
    continuation is re-emitted without continue=>1, which BNG2.pl would refuse
    beside t_start=>0 with the model time undefined."""
    spec = _parse(
        "simulate_nf({t_end=>2400,n_steps=>10})",
        "simulate_nf({t_end=>2400,n_steps=>10})",
        'simulate({method=>"ode",continue=>1,t_end=>5,n_steps=>5})',
    )
    first, again, cont = spec.steps
    assert spec.lossy == ()
    assert first.t_span == again.t_span == (0.0, 2400.0)
    assert cont.t_span == (0.0, 5.0)
    assert "continue" not in cont.extra
    assert parse_bngl_protocol(write_bngl_protocol(spec)).steps == spec.steps


@pytest.mark.parametrize("t_start", ["2400", "0"])
def test_a_continuation_naming_its_t_start_after_an_nf_run_is_refused(t_start):
    with pytest.raises(ConversionError, match=r"model time is undefined after an nf run"):
        _parse(
            "simulate_nf({t_end=>2400,n_steps=>10})",
            f'simulate({{method=>"ode",continue=>1,t_start=>{t_start},t_end=>2500,n_steps=>5}})',
            strict=False,
        )


@pytest.mark.parametrize(
    ("action", "message"),
    [
        ("simulate_nf({continue=>1,t_end=>20,n_steps=>10})", "NFsim does not take continue"),
        ('simulate({method=>"nf",continue=>1,t_end=>20,n_steps=>10})', "does not take continue"),
        ("simulate_nf({t_end=>10})", "needs n_steps of 1 or more"),
        ("simulate_nf({t_end=>10,n_output_steps=>10})", "needs n_steps of 1 or more"),
        ("simulate_nf({t_end=>10,n_steps=>0})", "needs n_steps of 1 or more"),
        ("simulate_nf({t_start=>10,t_end=>5,n_steps=>5})", r"needs t_end > t_start"),
        ("simulate_nf({n_steps=>5})", r"needs t_end;"),
    ],
    ids=[
        "continue",
        "method-nf-continue",
        "no-n_steps",
        "n_output_steps",
        "zero-steps",
        "t_end-before-t_start",
        "no-t_end",
    ],
)
def test_what_simulate_nf_refuses_is_refused(action, message):
    with pytest.raises(ConversionError, match=message):
        _parse(action, strict=False)


def test_simulate_nf_ignores_n_output_steps_beside_n_steps():
    (exp,) = _only("simulate_nf({t_end=>10,n_steps=>5,n_output_steps=>9})")
    assert exp.n_points == 6


def test_output_step_interval_is_lossy():
    with pytest.raises(ConversionError, match=r"output_step_interval writes rows"):
        _parse('simulate({method=>"ssa",t_end=>10,n_steps=>2,output_step_interval=>10})')


def test_simulate_nf_has_no_output_step_interval():
    """simulate_nf ignores it, and its rows are the n_steps grid."""
    (exp,) = _only("simulate_nf({t_end=>10,n_steps=>5,output_step_interval=>3})")
    assert (exp.t_span, exp.n_points) == ((0.0, 10.0), 6)


def test_a_run_whose_t_end_is_not_past_its_start_leaves_the_model_time():
    """BNG2.pl warns that such a run won't run, and the model time stays."""
    *_, last = _only(
        'simulate({method=>"ode",t_end=>10,n_steps=>10})',
        'simulate({method=>"ode",continue=>1,t_end=>5,n_steps=>5})',
        'simulate({method=>"ode",continue=>1,t_start=>10,t_end=>20,n_steps=>5})',
    )
    assert last.t_span == (10.0, 20.0)


def test_an_nf_scan_after_a_run_starts_only_its_first_point_there():
    """Each point of an nf scan starts at the model time, which the first
    point leaves at 0 (scaffold.bngl in the parity corpus)."""
    with pytest.raises(ConversionError, match=r"first point starts at the model time, 60"):
        _parse(
            'simulate({method=>"ode",t_end=>60,n_steps=>60})',
            'parameter_scan({method=>"nf",t_end=>100,n_steps=>10,parameter=>"k",'
            "par_min=>1,par_max=>2,n_scan_pts=>2})",
        )
    (exp,) = _only(
        'parameter_scan({method=>"nf",t_end=>100,n_steps=>10,parameter=>"k",'
        "par_min=>1,par_max=>2,n_scan_pts=>2})"
    )
    assert exp.t_span == (0.0, 100.0)


def test_nf_sample_times_are_lossy():
    with pytest.raises(ConversionError, match=r"refuses sample_times for NFsim"):
        _parse("simulate_nf({t_end=>10,sample_times=>[0,5,10]})")


def test_par_scan_vals_beside_par_min_and_par_max_without_n_scan_pts():
    """BNG2.pl takes par_min/par_max only with n_scan_pts as well."""
    (exp,) = _only(
        'parameter_scan({parameter=>"k",par_min=>1,par_max=>9,par_scan_vals=>[1,2,3],t_end=>10})'
    )
    assert (exp.scan_min, exp.scan_max, exp.scan_points) == (1.0, 3.0, 3)


@pytest.mark.parametrize(
    ("values", "expected"),
    [("[3,2,1]", (3.0, 1.0, 3, False)), ("[1,0.1,0.01]", (1.0, 0.01, 3, True))],
    ids=["linear", "log"],
)
def test_a_scan_without_reset_keeps_its_order(values, expected):
    """With reset_conc=>0 each point continues from the last, so a descending
    scan is held as par_min > par_max, which BNG2.pl runs as given."""
    spec = _parse(
        f'parameter_scan({{parameter=>"k",par_scan_vals=>{values},reset_conc=>0,t_end=>10}})'
    )
    (exp,) = spec.steps
    assert (exp.scan_min, exp.scan_max, exp.scan_points, exp.scan_log) == pytest.approx(expected)
    assert parse_bngl_protocol(write_bngl_protocol(spec)).steps == spec.steps


def test_an_unordered_scan_without_reset_is_lossy():
    with pytest.raises(ConversionError, match=r"in the order reset_conc=>0 runs them"):
        _parse('parameter_scan({parameter=>"k",par_scan_vals=>[3,1,2],reset_conc=>0,t_end=>10})')


def test_an_action_with_two_lossy_arguments_is_one_lossy_entry():
    with pytest.warns(ConversionWarning, match=r"sample_times give .*; par_scan_vals"):
        spec = _parse(
            'parameter_scan({parameter=>"k",par_scan_vals=>[1,2,7],sample_times=>[0,1,5]})',
            strict=False,
        )
    assert spec.lossy == ("parameter_scan",)
