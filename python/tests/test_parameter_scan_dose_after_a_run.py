"""parameter_scan over a parameter that seeds an initial condition, after a run.

Each point starts from the state at the call (or a named snapshot), and a
parameter write moves a species only while it sits on its initial condition
(Model.set_param, #141). After a run it does not: a scan over A0 = 5 and 40
started both points at the run's A = 6.07, for totals of 10 and 10 where a
fresh model gives 5 and 40, with no word (found on #1643). The scan keeps that
start, which a pre-equilibrated protocol relies on, and now says so, naming the
species and the two ways to dose them. Each remedy is run here.

The model: A -> B at k = 0.1 from A = A0*s, A0 = 10, s = 1, run to t = 5. The
snapshot is the state that run left, read off the model. A point that starts at
(a, b) holds A = a e^-k and B = b + a (1 - e^-k) one unit of time on.
"""

from __future__ import annotations

import math
import warnings

import bngsim
import numpy as np
import pytest

ANT = "model m\n A0 = 10; s = 1; k = 0.1; species A, B; A = A0*s; B = 0;\n J: A -> B; k*A;\nend"
# After the run to t = 5 a scan starts where it ended: one that set the clock
# back to 0 would draw main's warning for that too (issue #1701), which is not
# what this file asks about.
SCAN = {"t_span": (5, 6), "n_points": 2}
DROPS = r"parameter_scan over '{}' does not move the species it seeds \({}\)"


def _sim(src=ANT):
    pytest.importorskip("antimony")
    return bngsim.Simulator(bngsim.Model.from_antimony_string(src), method="ode")


def _after_a_run(src=ANT):
    """The Simulator after a run to t = 5, and the state that run left."""
    sim = _sim(src)
    sim.run(t_span=(0, 5), n_points=2)
    return sim, np.asarray(sim.model.get_state(), dtype=float)


def _expect(point, a0, b0, k=0.1):
    """A point that starts at (a0, b0) exactly and follows A -> B at k."""
    x = np.asarray(point.species)
    assert x[0, 0] == a0
    assert x[0, 1] == b0
    e = math.exp(-k)
    assert x[-1] == pytest.approx([a0 * e, b0 + a0 * (1 - e)], rel=1e-6)


def _quiet(call):
    """``call()``, with any UserWarning an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return call()


def test_a_dose_scan_after_a_run_says_the_dose_is_dropped():
    """Main ran this with no word. The points start where the run ended, as
    documented, and the warning names A, on the caller's line."""
    sim, snap = _after_a_run()
    with pytest.warns(UserWarning, match=DROPS.format("A0", "'A'")) as caught:
        points = sim.parameter_scan("A0", [5.0, 40.0, 10.0], **SCAN)
    (warning,) = [w for w in caught if "does not move" in str(w.message)]
    assert warning.filename == __file__
    assert "Model.reset() first" in str(warning.message)
    assert "on_point=lambda model, value: model.set_concentration(name, value)" in str(
        warning.message
    )
    for point in points:
        _expect(point, snap[0], snap[1])


def test_the_reset_the_warning_names_gives_the_dose_scan_of_a_fresh_model():
    """Remedy one, run: Model.reset() first starts each point at its dose with
    B at 0, totals 5 and 40, and there is nothing left to warn of."""
    sim, _ = _after_a_run()
    sim.model.reset()
    points = _quiet(lambda: sim.parameter_scan("A0", [5.0, 40.0], **SCAN))
    for dose, point in zip([5.0, 40.0], points, strict=True):
        _expect(point, dose, 0.0)


def test_the_hook_the_warning_names_doses_from_the_current_state():
    """Remedy two, run as written: the hook sets A at each point, B stays where
    the run left it, and a scan with a hook is not warned about."""
    sim, snap = _after_a_run()
    name = "A"
    points = _quiet(
        lambda: sim.parameter_scan(
            "A0",
            [5.0, 40.0],
            on_point=lambda model, value: model.set_concentration(name, value),
            **SCAN,
        )
    )
    for dose, point in zip([5.0, 40.0], points, strict=True):
        _expect(point, dose, snap[1])


def test_the_hook_sets_each_species_to_its_initial_condition_at_the_value():
    """D = 2*A0 is not the parameter itself: written with the value, the hook
    starts D at 5 and 40 where a fresh model starts it at 10 and 80. The message
    says to set each species to its initial condition at the point's value, 2*value
    for D, and the hook written so doses A and D as a fresh model does."""
    src = ANT.replace("species A, B;", "species A, B, D;").replace("B = 0;", "B = 0; D = 2*A0;")
    src = src.replace(" J: A -> B; k*A;\n", " J: A -> B; k*A;\n J2: D -> B; k*D;\n")
    sim, _ = _after_a_run(src)
    with pytest.warns(UserWarning, match=DROPS.format("A0", "'A', 'D'")) as caught:
        sim.parameter_scan("A0", [5.0, 40.0], **SCAN)
    (warning,) = [w for w in caught if "does not move" in str(w.message)]
    assert "set each in on_point to its initial condition at the point's value" in str(
        warning.message
    )
    assert "where that initial condition is the parameter itself" in str(warning.message)

    def dose(model, value):
        model.set_concentration("A", value)
        model.set_concentration("D", 2 * value)

    sim, _ = _after_a_run(src)
    points = _quiet(lambda: sim.parameter_scan("A0", [5.0, 40.0], on_point=dose, **SCAN))
    for value, point in zip([5.0, 40.0], points, strict=True):
        assert point.species[0, 0] == value
        assert point.species[0, 2] == 2 * value


def test_a_fresh_model_is_dosed_with_no_word():
    points = _quiet(lambda: _sim().parameter_scan("A0", [5.0, 40.0], **SCAN))
    for dose, point in zip([5.0, 40.0], points, strict=True):
        _expect(point, dose, 0.0)


def test_a_parameter_that_seeds_nothing_is_not_warned_about():
    """A scan over a rate constant moves no initial condition."""
    sim, snap = _after_a_run()
    ks = [0.1, 0.3]
    points = _quiet(lambda: sim.parameter_scan("k", ks, **SCAN))
    for k, point in zip(ks, points, strict=True):
        _expect(point, snap[0], snap[1], k=k)


def test_a_seeded_species_still_on_its_initial_condition_is_moved_and_not_named():
    """C = A0 is seeded and no reaction touches it: it still sits on its
    initial condition after the run, the write moves it, and only A is named."""
    src = ANT.replace("species A, B;", "species A, B, C;").replace("B = 0;", "B = 0; C = A0;")
    sim, _ = _after_a_run(src)
    with pytest.warns(UserWarning, match=DROPS.format("A0", "'A'")):
        points = sim.parameter_scan("A0", [5.0], **SCAN)
    assert points[0].species[0, 2] == 5.0


def test_a_derived_initial_condition_is_followed_to_the_parameter_it_reads():
    """A = A0*s: a scan over s is a dose over A too."""
    sim, snap = _after_a_run()
    with pytest.warns(UserWarning, match=DROPS.format("s", "'A'")):
        points = sim.parameter_scan("s", [0.5, 4.0], **SCAN)
    for point in points:
        _expect(point, snap[0], snap[1])


def test_a_named_snapshot_is_said_too():
    sim, snap = _after_a_run()
    sim.model.save_concentrations("eq")
    with pytest.warns(UserWarning, match=DROPS.format("A0", "'A'")):
        points = sim.parameter_scan("A0", [5.0], reset_to="eq", **SCAN)
    _expect(points[0], snap[0], snap[1])


def test_a_snapshot_taken_on_the_initial_conditions_is_dosed_with_no_word():
    """What is asked is where the points start, not where the model stands: a
    snapshot saved before the run holds A on its initial condition, so a scan
    reset to it after the run doses A, B at 0, and has nothing to warn of."""
    sim = _sim()
    sim.model.save_concentrations("fresh")
    sim.run(t_span=(0, 5), n_points=2)
    points = _quiet(lambda: sim.parameter_scan("A0", [5.0, 40.0], reset_to="fresh", **SCAN))
    for dose, point in zip([5.0, 40.0], points, strict=True):
        _expect(point, dose, 0.0)


def test_after_an_unlabeled_save_the_reason_is_the_saved_baseline():
    """save_concentrations() with no label makes the saved state the baseline,
    which no write moves: the same warning, with that reason."""
    sim, snap = _after_a_run()
    sim.model.save_concentrations()
    with pytest.warns(UserWarning, match=DROPS.format("A0", "'A'")) as caught:
        points = sim.parameter_scan("A0", [5.0, 40.0], **SCAN)
    assert "the saved state is the baseline" in str(caught[0].message)
    for point in points:
        _expect(point, snap[0], snap[1])
    _quiet(lambda: sim.parameter_scan("k", [0.2], **SCAN))


CHAIN_NET = """begin parameters
    1 A0 10
    2 T 2*A0
    3 U T+0
    4 k 0.1
end parameters
begin species
    1 A() U
    2 B() 0
end species
begin reactions
    1 1 2 k
end reactions
"""


def test_a_seed_two_parameters_deep_is_found(tmp_path):
    """A's initial condition is U, U = T and T = 2*A0. After a run the model
    says which initial conditions a write of A0 moves; after an unlabeled save
    none moves, and only following U to T to A0 finds A."""
    path = tmp_path / "chain.net"
    path.write_text(CHAIN_NET)
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    sim.run(t_span=(0, 5), n_points=2)
    with pytest.warns(UserWarning, match=DROPS.format("A0", r"'A\(\)'")):
        sim.parameter_scan("A0", [5.0], **SCAN)
    sim.model.save_concentrations()
    with pytest.warns(UserWarning, match=DROPS.format("A0", r"'A\(\)'")):
        sim.parameter_scan("A0", [5.0], **SCAN)


def test_the_message_names_three_species_and_counts_the_rest():
    species = ", ".join(f"S{i}" for i in range(5))
    inits = " ".join(f"S{i} = A0;" for i in range(5))
    decays = " ".join(f"J{i}: S{i} -> ; 0.1*S{i};" for i in range(5))
    src = f"model m\n A0 = 10; species {species}; {inits}\n {decays}\nend"
    sim, _ = _after_a_run(src)
    with pytest.warns(UserWarning, match=r"\('S0', 'S1', 'S2' and 2 more\)"):
        sim.parameter_scan("A0", [5.0], **SCAN)


def test_the_scan_leaves_the_model_as_it_found_it():
    sim, snap = _after_a_run()
    initial = list(sim.model._core.get_initial_state())
    with pytest.warns(UserWarning, match="does not move"):
        sim.parameter_scan("A0", [5.0, 40.0], **SCAN)
    assert list(sim.model.get_state()) == list(snap)
    assert list(sim.model._core.get_initial_state()) == initial
    assert sim.model.get_param("A0") == 10.0
    after = sim.run(t_span=(5, 6), n_points=2)  # continues from where the run ended
    assert after.species[-1, 0] == pytest.approx(snap[0] * math.exp(-0.1), rel=1e-6)


def test_a_continuation_scan_is_not_warned_about():
    """bifurcate starts each point where the last ended, by design."""
    sim, snap = _after_a_run()
    points = _quiet(lambda: sim.bifurcate("A0", [5.0, 40.0], **SCAN))
    assert points[0].species[0, 0] == snap[0]
    assert points[1].species[0, 0] == points[0].species[-1, 0]


@pytest.mark.parametrize("scanned", ["A0", "T", "U"])
def test_a_scan_over_a_derived_parameter_leaves_its_expression_live(tmp_path, scanned):
    """The warning asks the model which initial conditions a write moves, by
    writing the scanned parameter once. Written to a derived parameter, that
    overrides its expression until the value goes back (#188). The scan's reset
    puts it back: after a scan over A0, T or U, the model's parameters, which of
    them are expressions, its state and its initial conditions are as they were,
    and T and U still follow A0."""
    path = tmp_path / "chain.net"
    path.write_text(CHAIN_NET)
    model = bngsim.Model.from_net(str(path))
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0, 5), n_points=2)

    def found():
        return (
            [model.get_param(n) for n in model.param_names],
            list(model.param_is_expression),
            list(model.get_state()),
            list(model._core.get_initial_state()),
        )

    before = found()
    with pytest.warns(UserWarning, match="does not move the species it seeds"):
        sim.parameter_scan(scanned, [5.0, 40.0], **SCAN)
    assert found() == before
    model.set_param("A0", 3.0)
    assert (model.get_param("T"), model.get_param("U")) == (6.0, 6.0)


def test_asking_does_not_change_what_the_scan_runs():
    """The warning writes the scanned parameter once to ask what it seeds. The
    scan's tolerance is read before that, from the model as the scan found it:
    a fresh model at A0 = 1e-9, scanned at that value, is the run() of that
    model, bit for bit. Read after, the tolerance would be A's at twice its
    value, 2e-17 for 1e-17. (From A0 = 10 both are the scalar, and tell
    nothing.)"""
    src = ANT.replace("A0 = 10;", "A0 = 1e-9;")
    span = {"t_span": (0, 20), "n_points": 21}
    point = _quiet(lambda: _sim(src).parameter_scan("A0", [1e-9], **span))[0]
    alone = _sim(src).run(**span)
    assert np.array_equal(np.asarray(point.species), np.asarray(alone.species))


# ── Shapes the warning has to read right, each also run on main ──────────────


def test_a_stochastic_scan_is_asked_the_same_way():
    """Counts are rounded: a fresh model seeded at 10.5 is not "off its initial
    condition", and is dosed with no word; after a run the scan warns. SSA and
    PSA, in one case (the suite's count is near its bound)."""
    pytest.importorskip("antimony")
    src = ANT.replace("A0 = 10;", "A0 = 10.5;")
    for method, kw in (("ssa", {}), ("psa", {"poplevel": 100})):
        fresh = bngsim.Simulator(bngsim.Model.from_antimony_string(src), method=method, **kw)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            points = fresh.parameter_scan("A0", [5.0, 40.0], seed=1, **SCAN)
        assert not [w for w in caught if "does not move" in str(w.message)], method
        assert [p.species[0, 0] for p in points] == [5.0, 40.0], method
        ran = bngsim.Simulator(bngsim.Model.from_antimony_string(ANT), method=method, **kw)
        ran.run(t_span=(0, 5), n_points=2, seed=1)
        with pytest.warns(UserWarning, match=DROPS.format("A0", "'A'")):
            ran.parameter_scan("A0", [5.0, 40.0], seed=1, **SCAN)


def test_a_fixed_species_and_a_rule_target_are_not_named():
    """A $-fixed species seeded by A0 is not integrated, so it still sits on
    its initial condition after a run and the write moves it; a rule's target
    follows its rule. Neither is a dose the scan dropped."""
    fixed = ANT.replace("species A, B;", "species $F, A, B;").replace("B = 0;", "B = 0; F = A0;")
    sim, _ = _after_a_run(fixed.replace("A = A0*s;", "A = 10;"))
    points = _quiet(lambda: sim.parameter_scan("A0", [5.0, 40.0], **SCAN))
    assert [p.species[0, 0] for p in points] == [5.0, 40.0]
    rule = ANT.replace("species A, B;", "species A, B, R;").replace(
        "A = A0*s;", "A = 10; R := A0 + B;"
    )
    sim, _ = _after_a_run(rule)
    _quiet(lambda: sim.parameter_scan("A0", [5.0, 40.0], **SCAN))


def test_the_value_the_warning_asks_with():
    """A dose not yet given is 0, and twice 0 moves nothing: it is asked with
    1. A is made during the run, so it sits off its initial condition, and the
    scan over A0 warns. And a value the initial condition cannot take does not
    stop the scan: A = 1/(20 - A0) is asked with 2*A0 = 20, runs as on main,
    and warns of the dose it dropped."""
    made = (
        "model m\n A0 = 0; k = 0.1; species A, B; A = A0; B = 0;\n"
        " J0: -> A; 2; J: A -> B; k*A;\nend"
    )
    pole = ANT.replace("A = A0*s;", "A = 1/(20 - A0);")
    for src, values in ((made, [5.0, 40.0]), (pole, [5.0, 12.0])):
        sim, snap = _after_a_run(src)
        with pytest.warns(UserWarning, match=DROPS.format("A0", "'A'")):
            points = sim.parameter_scan("A0", values, **SCAN)
        assert [p.species[0, 0] for p in points] == [snap[0], snap[0]]
    # The scan's own values are asked too, and 5 and 40 would find A whatever
    # the first value asked: a scan of 0 alone leaves only that one.
    sim, snap = _after_a_run(made)
    with pytest.warns(UserWarning, match=DROPS.format("A0", "'A'")):
        points = sim.parameter_scan("A0", [0.0], **SCAN)
    assert points[0].species[0, 0] == snap[0]


def test_a_write_the_model_refuses_is_no_reason_to_stop_the_scan(monkeypatch):
    """The warning is advice: where the model refuses the value it asks with,
    it says nothing, and a scan that writes only the value the parameter holds
    runs as it did."""
    sim, snap = _after_a_run()
    write = type(sim.model).set_param

    def refuses_20(model, name, value, **kw):
        if name == "A0" and value == 20.0:
            raise ValueError("this model does not take A0 = 20")
        return write(model, name, value, **kw)

    monkeypatch.setattr(type(sim.model), "set_param", refuses_20)
    points = _quiet(lambda: sim.parameter_scan("A0", [10.0], **SCAN))
    _expect(points[0], snap[0], snap[1])


def test_a_sensitivity_scan_carries_what_it_carried():
    """Scan a rate constant after a run that leaves dx/ds pending. The warning
    asks its question before the first point and the reset puts the model
    back: the carried dA/ds at each point's start is the run's own."""
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(ANT), method="ode", sensitivity_params=["s"]
    )
    if not sim._codegen_provides_sens_rhs():
        pytest.skip("codegen compile unavailable: no compiled sensitivity RHS on this install")
    run = sim.run(t_span=(0, 5), n_points=2)
    carried = np.asarray(run.sensitivities)[-1, :, 0]
    points = _quiet(lambda: sim.parameter_scan("k", [0.2, 0.3], **SCAN))
    for point in points:
        assert np.array_equal(np.asarray(point.sensitivities)[0, :, 0], carried)


def test_a_sensitivity_scan_names_every_species_the_parameter_seeds():
    """A sensitivity scan carries the state's dx/dθ into each point, and a
    parameter write there moves no species, not even one still on its initial
    condition: C = A0 starts at 10 at A0 = 5 and 40. Asked as a plain scan
    asks, C looked moved and was not named, and with C the only species A0
    seeds there was no warning at all. Model.reset() first is no remedy here
    (a sensitivity scan from the initial conditions is refused), so the
    message names only the hook."""
    src = ANT.replace("species A, B;", "species A, B, C;").replace(
        "A = A0*s; B = 0;", "A = 10*s; B = 0; C = A0;"
    )
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(src), method="ode", sensitivity_params=["k"]
    )
    if not sim._codegen_provides_sens_rhs():
        pytest.skip("codegen compile unavailable: no compiled sensitivity RHS on this install")
    sim.run(t_span=(0, 5), n_points=2)
    with pytest.warns(UserWarning, match=DROPS.format("A0", "'C'")) as caught:
        points = sim.parameter_scan("A0", [5.0, 40.0], **SCAN)
    assert [p.species[0, 2] for p in points] == [10.0, 10.0]
    (warning,) = [w for w in caught if "does not move" in str(w.message)]
    assert "Model.reset() first" not in str(warning.message)
    assert "on_point=lambda model, value" in str(warning.message)


def test_a_sensitivity_scan_of_the_value_held_names_only_what_it_drops():
    """C = A0 sits on its initial condition, 10, and a sensitivity scan of
    A0 = 10 alone asks it for nothing: it starts at its dose. A = A0*s is off
    its initial condition after the run, and its dose of 10 is dropped. Naming
    every species the parameter seeds named C too."""
    src = ANT.replace("species A, B;", "species A, B, C;").replace("B = 0;", "B = 0; C = A0;")
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(src), method="ode", sensitivity_params=["k"]
    )
    if not sim._codegen_provides_sens_rhs():
        pytest.skip("codegen compile unavailable: no compiled sensitivity RHS on this install")
    sim.run(t_span=(0, 5), n_points=2)
    with pytest.warns(UserWarning, match=DROPS.format("A0", "'A'")) as caught:
        points = sim.parameter_scan("A0", [10.0], **SCAN)
    (warning,) = [w for w in caught if "does not move" in str(w.message)]
    assert "'C'" not in str(warning.message)
    assert points[0].species[0, 2] == 10.0


def test_an_initial_condition_flat_at_the_probe_is_asked_at_the_scan_values():
    """A = (A0 - 10)*(A0 - 20) + 3 is 3 at A0 = 10 and at 2*A0 = 20, so a write
    of 20 alone moves nothing; the scan's own values, 5 and 40, move it."""
    sim, snap = _after_a_run(ANT.replace("A = A0*s;", "A = (A0 - 10)*(A0 - 20) + 3;"))
    with pytest.warns(UserWarning, match=DROPS.format("A0", "'A'")):
        points = sim.parameter_scan("A0", [5.0, 40.0], **SCAN)
    assert [p.species[0, 0] for p in points] == [snap[0], snap[0]]


def _net(tmp_path, params, species="1 A() U\n2 B() 0"):
    path = tmp_path / "m.net"
    path.write_text(
        f"begin parameters\n{params}\nend parameters\nbegin species\n{species}\n"
        "end species\nbegin reactions\n1 1 2 k\nend reactions\n",
        encoding="utf-8",
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    sim.run(t_span=(0, 5), n_points=2)
    sim.model.save_concentrations()
    return sim


def test_the_saved_baseline_walk_reads_the_graph_as_the_model_does(tmp_path):
    """After an unlabeled save the species come from the parameter graph, which
    has to be read as the model reads it: a chain 1500 derived parameters deep
    (deeper than Python's recursion limit) is walked, not recursed, and the
    scan that ran on main still runs; the exponent of 1e-3 is a number, not a
    parameter named e; and a derived parameter overridden by a write (#188) no
    longer reads its expression, so A0 no longer seeds A through it."""
    chain = "\n".join(f"{i + 4} P{i} {'A0' if i == 0 else f'P{i - 1}'}+0" for i in range(1500))
    sim = _net(tmp_path, f"1 A0 10\n2 k 0.1\n3 Z 1\n{chain}", "1 A() P1499\n2 B() 0")
    with pytest.warns(UserWarning, match=DROPS.format("A0", r"'A\(\)'")):
        sim.parameter_scan("A0", [5.0], **SCAN)
    _quiet(lambda: sim.parameter_scan("Z", [2.0], **SCAN))

    sim = _net(tmp_path, "1 A0 10\n2 e 3\n3 k 0.1\n4 U 1e-3*A0")
    _quiet(lambda: sim.parameter_scan("e", [5.0], **SCAN))
    with pytest.warns(UserWarning, match=DROPS.format("A0", r"'A\(\)'")):
        sim.parameter_scan("A0", [5.0], **SCAN)

    path = tmp_path / "chain.net"
    path.write_text(CHAIN_NET, encoding="utf-8")
    model = bngsim.Model.from_net(str(path))
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0, 5), n_points=2)
    model.set_param("T", 7.0)  # T no longer follows A0; U still follows T
    model.save_concentrations()
    _quiet(lambda: sim.parameter_scan("A0", [5.0], **SCAN))
    with pytest.warns(UserWarning, match=DROPS.format("T", r"'A\(\)'")):
        sim.parameter_scan("T", [5.0], **SCAN)


def test_a_compartment_size_is_said_alike_after_a_run_and_after_a_save():
    """A size write re-divides an amount-declared species' initial condition,
    so a scan over the size is a dose: the warning names A after a run, and
    after an unlabeled save, where the size route is not a parameter
    reference, alike."""
    src = (
        "model m\n compartment c = 2; substanceOnly species A in c; species B in c;\n"
        " A = 10; B = 0; k = 0.1;\n J: A -> B; k*A;\nend"
    )
    for save in (False, True):
        sim, snap = _after_a_run(src)
        if save:
            sim.model.save_concentrations()
        with pytest.warns(UserWarning, match=DROPS.format("c", "'A'")):
            points = sim.parameter_scan("c", [1.0, 4.0], **SCAN)
        assert [p.species[0, 0] for p in points] == [snap[0], snap[0]]
