"""A derivative that reads the rateOf of a species whose own derivative reads a
rateOf (lanl/bngsim#991's family, found by the twelfth review of #950's fix).

`S' = rateOf(A)` is exact after one probe of the right-hand side: the probe
gives dA/dt, and the evaluation after it reads that. `T' = k3*rateOf(S)` reads
dS/dt, which is exact only once dA/dt has been published, a probe later. With
one probe T read dS/dt as the call before had left it: T(1) = 0.44 for 0.216
at the default tolerances (rtol and atol 1e-8), 1.69 at rtol 1e-6, and right
only with atol at 1e-10 or under.
The generated right-hand side, which starts every call from a zeroed buffer,
read it as zero, and T never moved; a run with sensitivity columns integrates
with that one, and returned T = 0 with every dT/dp = 0.

The oracle is the closed form. B' = -k2*B from 1, A' = k1*B, so
dA/dt = k1*exp(-k2*t), S = k1*(1 - exp(-k2*t))/k2 and T = k3*S.
libRoadRunner 2.9.2 returns the same.
"""

import math

import bngsim
import numpy as np
import pytest

_K1, _K2, _K3 = 1.0, 2.0, 0.5
_NESTED = (
    "species A = 0, B = 1, S = 0, T = 0; B' = -k2*B; A' = k1*B;"
    " S' = rateOf(A); T' = k3*rateOf(S); k1 = 1; k2 = 2; k3 = 0.5; q = 0;\n"
)
_S1 = _K1 * (1.0 - math.exp(-_K2)) / _K2
_T1 = _K3 * _S1


def _last(result, name: str) -> float:
    return float(np.asarray(result.species)[-1, list(result.species_names).index(name)])


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "generated"])
@pytest.mark.parametrize("rows", [3, 101])
@pytest.mark.parametrize(
    "tolerances", [{}, {"rtol": 1e-6, "atol": 1e-9}], ids=["default", "loose"]
)
@pytest.mark.parametrize("event", ["", "Z: at (time >= 5): q = 1;\n"], ids=["alone", "an event"])
def test_a_flow_on_a_nested_rateof_integrates_to_the_closed_form(codegen, rows, tolerances, event):
    model = bngsim.Model.from_antimony_string(_NESTED + event)
    sim = bngsim.Simulator(model, method="ode", codegen=codegen)
    r = sim.run(sample_times=list(np.linspace(0.0, 1.0, rows)), **tolerances)
    assert _last(r, "S") == pytest.approx(_S1, rel=1e-5)
    assert _last(r, "T") == pytest.approx(_T1, rel=1e-5)


def test_a_run_with_sensitivity_columns_integrates_it_and_its_columns():
    """T = k3*k1*(1 - exp(-k2))/k2: dT/dk3 = S, dT/dk1 = k3*S/k1, and
    dT/dk2 = k3*k1*(k2*exp(-k2) - (1 - exp(-k2)))/k2^2."""
    model = bngsim.Model.from_antimony_string(_NESTED)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1", "k2", "k3"])
    r = sim.run(sample_times=[0.0, 0.5, 1.0], rtol=1e-10, atol=1e-12)
    assert _last(r, "T") == pytest.approx(_T1, rel=1e-7)
    columns = np.asarray(r.sensitivities)[-1, list(r.species_names).index("T")]
    d_k2 = _K3 * _K1 * (_K2 * math.exp(-_K2) - (1.0 - math.exp(-_K2))) / _K2**2
    assert columns == pytest.approx([_K3 * _S1 / _K1, d_k2, _S1], rel=1e-6)


def test_a_clone_and_a_batch_entry_integrate_it_as_the_model_does():
    """What reads which rateOf is asked of the model once and copied to its
    clones (each clone asked for itself, every rate law parsed again on the
    first run of each entry of a batch). A clone's copy says what the
    model's does."""
    model = bngsim.Model.from_antimony_string(_NESTED)
    twin = model.clone()
    for one in (model, twin):
        r = bngsim.Simulator(one, method="ode").run(sample_times=[0.0, 0.5, 1.0])
        assert _last(r, "T") == pytest.approx(_T1, rel=1e-5)
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(_NESTED), method="ode")
    batch = sim.run_batch(
        t_span=(0.0, 1.0), n_points=3, params=[{"k3": 0.5}, {"k3": 1.0}], num_processors=1
    )
    assert [_last(r, "T") for r in batch] == pytest.approx([_T1, 2.0 * _T1], rel=1e-5)


def test_a_rateof_that_nothing_reads_the_rateof_of_is_one_probe_deep():
    """`Y' = rateOf(A)` with no rateOf(Y) anywhere: exact as it was."""
    text = "species A = 0, B = 1, Y = 0; B' = -k2*B; A' = k1*B; Y' = rateOf(A); k1 = 1; k2 = 2;\n"
    r = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode").run(
        sample_times=[0.0, 0.5, 1.0]
    )
    assert _last(r, "Y") == pytest.approx(_S1, rel=1e-6)


# ── What the thirteenth review found ─────────────────────────────────
def _chain(n: int, readers: bool = False) -> str:
    """Y1' = rateOf(A), and each Y after it a rate rule on the rateOf of the
    one before: every one of them is S."""
    text = "species A = 0, B = 1, q = -1; B' = -k2*B; A' = k1*B; k1 = 1; k2 = 2;\n"
    for i in range(1, n + 1):
        text += f"species Y{i} = 0; Y{i}' = rateOf({'A' if i == 1 else f'Y{i - 1}'});\n"
    if readers:
        text += (
            "G: at (time >= 0.5), priority = 10: B = 5;\n"
            f"H: at (time >= 0.5), priority = 1, fromTrigger = false: q = rateOf(Y{n});\n"
        )
    return text


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "generated"])
@pytest.mark.parametrize("n", [9, 10, 12])
def test_a_chain_of_nested_rateofs_is_as_deep_as_it_is_long(n, codegen):
    """The depth was counted to eight for a chain as for a ring: the tenth
    rate rule of a chain read its rateOf a probe behind (0.087 for 0.562 with
    the reviewer's constants; zero when generated), where the ninth was exact."""
    model = bngsim.Model.from_antimony_string(_chain(n))
    assert model._core.codegen_data()["rateof_flow_nesting"] == n - 1
    r = bngsim.Simulator(model, method="ode", codegen=codegen).run(sample_times=[0.0, 0.5, 1.0])
    assert _last(r, f"Y{n}") == pytest.approx(_S1, rel=1e-5)


@pytest.mark.parametrize("n", [8, 10])
def test_an_event_value_reads_the_end_of_a_chain_as_it_is_after_a_write(n):
    """G sets B = 5 at 0.5 and H, which executes after it on the same
    instant, stores rateOf(Yn) = k1*B = 5: one probe for each link of the
    chain and one more."""
    model = bngsim.Model.from_antimony_string(_chain(n, readers=True))
    assert model._core.codegen_data()["rateof_nesting"] == n
    r = bngsim.Simulator(model, method="ode").run(sample_times=[0.0, 0.5, 1.0])
    assert _last(r, "q") == pytest.approx(5.0, rel=1e-9)


def test_a_rateof_nothing_takes_the_rateof_of_is_counted_as_one_probe():
    """`Y' = rateOf(A)`: what is read is rateOf(A), which is exact after one
    probe. Counted over every derivative that reads a rateOf it was two, at
    every row, stop and event. With a trigger on rateOf(Y) it is two."""
    plain = "species A = 0, B = 1, Y = 0; B' = -k2*B; A' = k1*B; Y' = rateOf(A); k1 = 1; k2 = 2;\n"
    data = bngsim.Model.from_antimony_string(plain)._core.codegen_data()
    assert (data["rateof_flow_nesting"], data["rateof_nesting"]) == (0, 0)
    read = plain + "n = 0;\nE: at (rateOf(Y) < 0.5): n = n + 1;\n"
    data = bngsim.Model.from_antimony_string(read)._core.codegen_data()
    assert (data["rateof_flow_nesting"], data["rateof_nesting"]) == (0, 1)


def test_a_clone_made_while_the_model_is_first_evaluated_has_what_was_read_of_it():
    """`Model.rhs` releases the interpreter lock, and the first evaluation of
    a model reads off its rate laws which rateOf each derivative reads. A
    clone asks its original for the same. With the lists marked known before
    they were filled, a clone made in between took empty ones: nesting 0,
    and dY2/dt = 0."""
    import threading

    n = 400
    text = (
        "species "
        + ", ".join(f"X{i} = {1 + 0.01 * (i % 5)}" for i in range(n))
        + ", Y = 0, Y2 = 0;\nk = 0.3;\n"
        + "".join(
            f"J{i}: X{i} -> X{(i + 1) % n}; k*X{i}/(1 + 0.1*X{(i + 7) % n});\n" for i in range(n)
        )
        + "Y' = rateOf(X5); Y2' = rateOf(Y);\n"
    )
    fresh = bngsim.Model.from_antimony_string(text)
    y0 = np.asarray(fresh._core.get_state(), dtype=float)
    want = np.asarray(fresh.rhs(y0))
    for _ in range(3):
        model = bngsim.Model.from_antimony_string(text)
        evaluating = threading.Thread(target=lambda model=model: model.rhs(y0))
        evaluating.start()
        twin = model.clone()
        evaluating.join()
        assert twin._core.codegen_data()["rateof_flow_nesting"] == 1
        assert np.asarray(twin.rhs(y0)) == pytest.approx(want, rel=1e-12)


# ── What the fourteenth review found ─────────────────────────────────────────
# Built with ModelBuilder: Antimony and libSBML refuse a ring of rate rules, so
# nothing above went through one.


def _builder():
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.enable_rateof()
    return b


def _decay_into_a(b):
    """A' = k1*B, B' = -k2*B from 1: dA/dt = k1*exp(-k2*t)."""
    a = b.add_species("A", 0.0)
    bb = b.add_species("B", 1.0)
    b.add_parameter("k1", _K1)
    b.add_parameter("k2", _K2)
    b.add_observable("Bo", [(bb, 1.0)])
    b.add_function("fA", "k1*Bo")
    b.add_function("fB", "k2*Bo")
    b.add_reaction([], [a], "functional", "fA")
    b.add_reaction([bb], [], "functional", "fB", 1.0, False)


def _chain_with_a_catalyst(n: int, lo: int, hi: int, first: int) -> bngsim.Model:
    """C0' = rateOf(A), Ci' = rateOf(C(i-1)), and `C_lo + S -> C_lo + P` at
    kc*rateOf(C_hi), with C_first declared before the rest."""
    b = _builder()
    order = [first] + [i for i in range(n + 1) if i != first]
    c = {i: b.add_species(f"C{i}", 0.0) for i in order}
    s = b.add_species("S", 5.0)
    p = b.add_species("P", 0.0)
    _decay_into_a(b)
    b.add_parameter("kc", 0.7)
    for i in range(n + 1):
        b.add_function(f"f{i}", "rate_of__A" if i == 0 else f"rate_of__C{i - 1}")
        b.add_reaction([], [c[i]], "functional", f"f{i}")
    b.add_function("fc", f"kc*rate_of__C{hi}")
    b.add_reaction([c[lo], s], [c[lo], p], "functional", "fc", 1.0, False)
    return bngsim.Model(_core=b.build())


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "generated"])
@pytest.mark.parametrize("first", [0, 4, 6, 11])
def test_a_catalyst_closes_no_ring(first, codegen):
    """`C3 + S -> C3 + P` at kc*rateOf(C10): C3 stands on both sides, and the
    reaction does not change it. Its derivative holds nothing of the rate
    (zero times it, and on a probe not that), so it reads nothing of the rate
    law, and no ring runs from C10 back to C3 in the chain of fifteen. While
    the right-hand side applied a reaction's two sides one after the other it
    did hold it, and was given the reads; the ring that was none then had to
    be counted deep enough, and was not: nine probes for fifteen by which
    species was declared first, C12(1) = 2.49 for 0.562 (0 from generated
    code)."""
    model = _chain_with_a_catalyst(15, 3, 10, first)
    assert model._core.rateof_ring_refusal() == ""
    assert model._core.codegen_data()["rateof_flow_nesting"] == 15
    r = bngsim.Simulator(model, method="ode", codegen=codegen).run(sample_times=[0.0, 0.5, 1.0])
    for name in ("C3", "C10", "C12", "C15"):
        assert _last(r, name) == pytest.approx(_S1, rel=2e-5)
    assert _last(r, "P") == pytest.approx(0.7 * _S1, rel=2e-5)


def _catalyst_at_a_huge_rate(law: str = "1/(1e-17 + rate_of__A)") -> bngsim.Model:
    """0 -> E at 1; E + S -> E + P at *law*; Y' = rateOf(E)."""
    b = _builder()
    e = b.add_species("E", 2.0)
    s = b.add_species("S", 5.0)
    p = b.add_species("P", 0.0)
    y = b.add_species("Y", 0.0)
    _decay_into_a(b)
    b.add_function("fc", law)
    b.add_function("one", "1")
    b.add_function("fY", "rate_of__E")
    b.add_reaction([], [e], "functional", "one")
    b.add_reaction([e, s], [e, p], "functional", "fc", 1.0, False)
    b.add_reaction([], [y], "functional", "fY")
    return bngsim.Model(_core=b.build())


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "generated"])
@pytest.mark.parametrize(
    "law", ["1/(1e-17 + rate_of__A)", "1/rate_of__A"], ids=["huge", "infinite"]
)
def test_a_catalyst_reads_nothing_of_the_rate_law(law, codegen):
    """E stands on both sides of a reaction whose rate is 1/(1e-17 +
    rateOf(A)), or 1/rateOf(A). On the first probe, where rateOf(A) is not
    there yet, that rate is 1e17, or infinite, and it is the model's rate on
    no pass but the last. dE/dt is 1 on every pass, and `Y' = rateOf(E)` is
    right after one probe: no nesting.

    With the two sides applied one after the other, dE/dt was (1 - 1e17) +
    1e17 = 0 on that probe (inf - inf, not a number, with the second law),
    and Y(1) came out 0 for 1 from generated code wherever E was given no
    read of the rate law."""
    model = _catalyst_at_a_huge_rate(law)
    assert model._core.codegen_data()["rateof_flow_nesting"] == 0
    names = list(model.species_names)
    y0 = np.asarray(model._core.get_state(), dtype=float)
    first = np.asarray(model.rhs(y0))
    assert first[names.index("E")] == 1.0
    assert first[names.index("Y")] == pytest.approx(1.0, rel=1e-9)
    r = bngsim.Simulator(_catalyst_at_a_huge_rate(law), method="ode", codegen=codegen).run(
        sample_times=[0.0, 0.5, 1.0]
    )
    assert _last(r, "Y") == pytest.approx(1.0, rel=1e-6)
    assert _last(r, "E") == pytest.approx(3.0, rel=1e-6)
    # P' = 1/rateOf(A) = exp(k2*t)/k1
    assert _last(r, "P") == pytest.approx((math.exp(_K2) - 1.0) / (_K1 * _K2), rel=1e-5)


def test_a_catalyst_in_a_right_hand_side_emitted_in_blocks(monkeypatch):
    """A large model's reactions are emitted as blocks, functions of their
    own, which do not see the pass the right-hand side is on: a block with a
    statement of the last pass alone is handed which pass it is."""
    from bngsim._codegen import prepare_model_codegen_source

    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "1")
    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", "1")
    src = prepare_model_codegen_source(_catalyst_at_a_huge_rate("1/rate_of__A"))
    assert "bngsim-codegen: chunked" in src
    assert "double* ydot, int _rateof_last)" in src
    assert "if (_rateof_last) ydot[0] -= 0.0 * (rate);" in src
    r = bngsim.Simulator(_catalyst_at_a_huge_rate("1/rate_of__A"), method="ode", codegen=True).run(
        sample_times=[0.0, 0.5, 1.0]
    )
    assert _last(r, "Y") == pytest.approx(1.0, rel=1e-6)
    assert _last(r, "E") == pytest.approx(3.0, rel=1e-6)
    assert _last(r, "P") == pytest.approx((math.exp(_K2) - 1.0) / (_K1 * _K2), rel=1e-5)


def _catalyst_whose_rate_is_read(variant: str, catalyst: bool = True) -> bngsim.Model:
    """`E -> 0` at kd*E, so rateOf(E) = -0.3*E, and `E + S -> E + P` at a law
    that is 1/rateOf(A) from t = 0.5 on (*variant* "event": an event switches
    it on, with dA/dt; "clock": the time does; "always": from the start). A
    row reads rateOf(E), and an event T, true from the start, counts in F how
    often `rateOf(E) < -0.1` rises."""
    b = _builder()
    a = b.add_species("A", 0.0)
    e = b.add_species("E", 2.0)
    s = b.add_species("S", 5.0)
    p = b.add_species("P", 0.0)
    g = b.add_species("G", 0.0)
    f = b.add_species("F", 0.0)
    tf = b.add_species("TF", -1.0)
    b.add_parameter("kd", 0.3)
    b.add_observable("Eo", [(e, 1.0)])
    b.add_observable("Go", [(g, 1.0)])
    b.add_observable("OF", [(f, 1.0)])
    if variant == "event":
        b.add_function("fa", "2*Go")
        b.add_function("fJ", "if(Go > 0.5, 1/rate_of__A, 0)")
        b.add_event("U", "time() >= 0.5", [(g, "1")], initial_value=False)
    elif variant == "clock":
        b.add_function("fa", "if(time() >= 0.5, 2, 0)")
        b.add_function("fJ", "if(time() >= 0.5, 1/rate_of__A, 0)")
    else:
        b.add_function("fa", "2")
        b.add_function("fJ", "1/rate_of__A")
    b.add_function("fd", "kd*Eo")
    b.add_function("rE", "rate_of__E")
    b.add_reaction([], [a], "functional", "fa")
    b.add_reaction([e], [], "functional", "fd", 1.0, False)
    b.add_reaction(
        [e, s] if catalyst else [s], [e, p] if catalyst else [p], "functional", "fJ", 1.0, False
    )
    b.add_event("T", "rate_of__E < -0.1", [(f, "OF + 1"), (tf, "time()")], initial_value=True)
    return bngsim.Model(_core=b.build())


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "generated"])
@pytest.mark.parametrize("variant", ["always", "event", "clock"])
def test_what_a_row_and_a_trigger_read_as_the_rate_of_a_catalyst(variant, codegen):
    """The rates a row, a trigger and an event value read are the last of the
    probes `refresh_rateof_derivs` makes, and that probe's rate laws read what
    the one before left: for a species nothing is counted a probe deep for,
    the call before. A catalyst is such a species. Given zero times a law
    that is infinite for that alone (1/rateOf(A) with the dA/dt of before the
    switch, or of a model first asked), rateOf(E) was not a number in the row,
    and the trigger on it, true all along, fired on the instant of the other
    event: F = 1 at t = 0.5."""
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    want = [-0.6 * math.exp(-0.3 * t) for t in times]
    for catalyst in (True, False):
        model = _catalyst_whose_rate_is_read(variant, catalyst)
        assert model._core.codegen_data()["rateof_nesting"] == 0
        r = bngsim.Simulator(model, method="ode", codegen=codegen).run(sample_times=times)
        rows = np.asarray(r.raw_expressions)[:, list(r.raw_expression_names).index("rE")]
        assert rows == pytest.approx(want, rel=1e-6)
        assert _last(r, "F") == 0.0 and _last(r, "TF") == -1.0
        assert _last(r, "P") == pytest.approx(0.5 if variant == "always" else 0.25, rel=1e-6)


def test_a_rate_that_is_not_finite_on_the_last_pass_still_fails_the_run():
    """What a catalyst is told of a rate that is not finite, it is told by
    the pass that is the model's: `E + S -> E + P` at 1/(rateOf(A) - k1),
    which divides by zero at t = 0 for good."""
    with pytest.raises(bngsim.SimulationError, match="non-finite"):
        bngsim.Simulator(_catalyst_at_a_huge_rate("1/(rate_of__A - k1)"), method="ode").run(
            sample_times=[0.0, 0.5, 1.0]
        )


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "generated"])
@pytest.mark.parametrize("sides", ["E -> E", "0 -> 0"])
def test_the_rate_law_of_a_reaction_that_changes_nothing_is_still_a_column(sides, codegen):
    """`fc = rateOf(S)` over `S' = rateOf(A)` as the rate law of `E -> E`, or
    of a reaction with no species at all: no derivative holds it, and it is
    a column of the result all the same, read at each row after the probes
    its nesting takes. Credited to no one, it was read a probe short: 0 at
    the first row and each later row a few percent off (from generated code,
    each row the row before)."""
    b = _builder()
    s = b.add_species("S", 0.0)
    e = b.add_species("E", 2.0)
    _decay_into_a(b)
    b.add_function("fS", "rate_of__A")
    b.add_function("fc", "rate_of__S")
    b.add_reaction([], [s], "functional", "fS")
    both = [e] if sides == "E -> E" else []
    b.add_reaction(both, both, "functional", "fc", 1.0, False)
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    sim = bngsim.Simulator(bngsim.Model(_core=b.build()), method="ode", codegen=codegen)
    r = sim.run(sample_times=times)
    column = np.asarray(r.expressions)[:, list(r.expression_names).index("fc")]
    assert column == pytest.approx([_K1 * math.exp(-_K2 * t) for t in times], rel=1e-5)


def _chain_on_a_ring(n: int, chain_first: bool) -> bngsim.Model:
    """X' = 0.5*rateOf(Y) + 1, Y' = 0.5*rateOf(X) + 1 (so X' = Y' = 2), and
    Z1' = rateOf(X), Zi' = rateOf(Z(i-1))."""
    b = _builder()
    z = {}
    if chain_first:
        z = {i: b.add_species(f"Z{i}", 0.0) for i in range(1, n + 1)}
    x = b.add_species("X", 0.0)
    y = b.add_species("Y", 0.0)
    if not chain_first:
        z = {i: b.add_species(f"Z{i}", 0.0) for i in range(1, n + 1)}
    b.add_function("fX", "0.5*rate_of__Y + 1")
    b.add_function("fY", "0.5*rate_of__X + 1")
    b.add_reaction([], [x], "functional", "fX")
    b.add_reaction([], [y], "functional", "fY")
    for i in range(1, n + 1):
        b.add_function(f"fZ{i}", "rate_of__X" if i == 1 else f"rate_of__Z{i - 1}")
        b.add_reaction([], [z[i]], "functional", f"fZ{i}")
    return bngsim.Model(_core=b.build())


@pytest.mark.parametrize("chain_first", [False, True], ids=["ring first", "chain first"])
def test_a_ring_of_rates_on_each_other_is_refused_by_name(chain_first):
    """`X' = 0.5*rateOf(Y) + 1` and `Y' = 0.5*rateOf(X) + 1` define each rate
    by the other at the same instant: an algebraic loop, which no number of
    probes reads. Taken as eight deep, it was integrated at its fixed point
    after eight passes from nothing (2 - 2^-8 here), and before that the top
    of a chain above it read the ring after two passes or three: Z8(1) = 1.5
    or 1.75 for 2, by the order of declaration. It is refused by name, as
    libRoadRunner, Antimony and libSBML refuse it."""
    from bngsim._codegen import prepare_model_codegen_source

    model = _chain_on_a_ring(8, chain_first)
    why = model._core.rateof_ring_refusal()
    assert "X and Y" in why and "lanl/bngsim#991" in why
    assert "d(X)/dt reads rateOf(Y)" in why and "d(Y)/dt reads rateOf(X)" in why
    for codegen in (False, True):
        with pytest.raises(bngsim.ModelError, match="lanl/bngsim#991"):
            bngsim.Simulator(_chain_on_a_ring(8, chain_first), method="ode", codegen=codegen)
    with pytest.raises(bngsim.ModelError, match="X and Y"):
        model.rhs(np.zeros(model.n_species))
    with pytest.raises(bngsim.ModelError, match="X and Y"):
        model.jacobian(np.zeros(model.n_species))
    # No right-hand side is emitted for it.
    assert prepare_model_codegen_source(model) is None
    with pytest.raises(RuntimeError, match="lanl/bngsim#991"):
        model._core.compute_derivs(0.0, np.zeros(model.n_species))


@pytest.mark.parametrize(
    "reads, ring",
    [
        ({0: [0]}, [0]),  # a species on its own rateOf
        ({0: [1], 1: [2], 2: [0]}, [0, 1, 2]),  # a ring of three
        ({0: [1, 2], 1: [0], 2: [0], 3: [0]}, [0, 1, 2]),  # two rings through one species
        ({0: [1], 1: [0], 2: [0, 1], 3: [2, 0]}, [0, 1]),  # a ring that another species reads
        ({i: [(i + 1) % 10] for i in range(10)}, list(range(10))),  # a ring of ten
        ({0: [1], 1: [0, 2], 2: [3], 3: []}, [0, 1]),  # a ring that reads a rate rule
        ({0: [], 1: [0], 2: [3], 3: [2]}, [2, 3]),  # a ring beside a chain
    ],
    ids=["self", "three", "two rings", "one above", "ten", "one below", "beside"],
)
def test_any_ring_of_rate_rules_is_refused_and_its_species_named(reads, ring):
    n = max(4, len(reads))
    b = _builder()
    species = [b.add_species(f"R{i}", 0.0) for i in range(n)]
    for i in range(n):
        terms = [f"0.25*rate_of__R{j}" for j in reads.get(i, [])]
        b.add_function(f"g{i}", " + ".join([*terms, "1"]))
        b.add_reaction([], [species[i]], "functional", f"g{i}")
    model = bngsim.Model(_core=b.build())
    why = model._core.rateof_ring_refusal()
    assert "lanl/bngsim#991" in why
    for i in ring[:6]:
        assert f"R{i}" in why
    for i in set(range(n)) - set(ring):
        assert f"R{i}" not in why.replace("rateOf(", "")
    if len(ring) == 1:
        assert "in terms of itself" in why and "d(R0)/dt reads rateOf(R0)" in why
    if len(ring) > 6:
        assert f"and {len(ring) - 6} more" in why
    with pytest.raises(bngsim.ModelError, match="algebraic loop"):
        bngsim.Simulator(model, method="ode")


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "generated"])
@pytest.mark.parametrize("shape", ["pair", "itself", "product"])
def test_a_species_that_is_held_fixed_closes_no_ring(shape, codegen):
    """A fixed species has a derivative of zero whatever its reactions add to
    it, and reads nothing. Given the reads of the reactions that would change
    it, `$X` made at a rate on rateOf(Y) beside `Y' = g*rateOf(X) + 1` was a
    ring, and the model was refused for it where Y' is 1."""
    b = _builder()
    b.add_parameter("g", 0.5)
    x = b.add_species("X", 2.0, True)
    if shape == "pair":
        y = b.add_species("Y", 0.0)
        b.add_function("fX", "g*rate_of__Y + 1")
        b.add_function("fY", "g*rate_of__X + 1")
        b.add_reaction([], [x], "functional", "fX")
        b.add_reaction([], [y], "functional", "fY")
        name, want = "Y", 1.0
    elif shape == "itself":
        y = b.add_species("Y", 0.0)
        b.add_function("fX", "rate_of__X + 1")
        b.add_function("fY", "rate_of__X + 3")
        b.add_reaction([], [x], "functional", "fX")
        b.add_reaction([], [y], "functional", "fY")
        name, want = "Y", 3.0
    else:
        s = b.add_species("S", 5.0)
        b.add_observable("So", [(s, 1.0)])
        b.add_function("fJ", "g*So*(1 + rate_of__X)")
        b.add_reaction([s], [x], "functional", "fJ", 1.0, False)
        name, want = "S", 5.0 * math.exp(-0.5)
    model = bngsim.Model(_core=b.build())
    assert model._core.rateof_ring_refusal() == ""
    r = bngsim.Simulator(model, method="ode", codegen=codegen).run(sample_times=[0.0, 0.5, 1.0])
    assert _last(r, "X") == 2.0
    assert _last(r, name) == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize(
    "reads, depth",
    [
        ({0: [], 1: [0], 2: [1], 3: [2]}, 3),  # a chain
        ({0: [], 1: [0], 2: [0, 1], 3: [0, 1, 2]}, 3),  # every one on all before it
        ({0: [], 1: [], 2: [0, 1], 3: [2]}, 2),  # two feet
    ],
    ids=["chain", "triangle", "two feet"],
)
def test_a_model_with_no_ring_is_counted_as_it_was(reads, depth):
    b = _builder()
    species = [b.add_species(f"R{i}", 0.0) for i in range(4)]
    for i in range(4):
        terms = [f"0.25*rate_of__R{j}" for j in reads.get(i, [])]
        b.add_function(f"g{i}", " + ".join([*terms, "1"]))
        b.add_reaction([], [species[i]], "functional", f"g{i}")
    b.add_function("top", "rate_of__R3")
    b.add_species("T", 0.0)
    b.add_reaction([], [4], "functional", "top")
    model = bngsim.Model(_core=b.build())
    assert model._core.rateof_ring_refusal() == ""
    assert model._core.codegen_data()["rateof_flow_nesting"] == depth


def test_a_chain_of_many_thousand_rate_rules_is_walked_on_a_small_stack():
    """One call for each link: 7,000 rate rules, each on the rateOf of the
    one before, ended the process on a thread with half a megabyte of stack
    (120,000 did on the main thread)."""
    import threading

    n = 12_000
    b = _builder()
    c = {i: b.add_species(f"C{i}", 0.0) for i in range(n, -1, -1)}
    a = b.add_species("A", 0.0)
    b.add_function("fA", "1")
    b.add_reaction([], [a], "functional", "fA")
    for i in range(n + 1):
        b.add_function(f"f{i}", "rate_of__A" if i == 0 else f"rate_of__C{i - 1}")
        b.add_reaction([], [c[i]], "functional", f"f{i}")
    model = bngsim.Model(_core=b.build())
    out = {}
    before = threading.stack_size(512 * 1024)
    try:
        thread = threading.Thread(
            target=lambda: out.update(flow=model._core.codegen_data()["rateof_flow_nesting"])
        )
        thread.start()
        thread.join()
    finally:
        threading.stack_size(before)
    assert out["flow"] == n


def _table_on_a_rateof(call: str, table: str = "kf") -> bngsim.Model:
    """S' = rateOf(A); ix := rateOf(S); the identity table over ix; T' = 2*<call>."""
    b = _builder()
    s = b.add_species("S", 0.0)
    t = b.add_species("T", 0.0)
    _decay_into_a(b)
    b.add_function("fS", "rate_of__A")
    b.add_parameter("ix", 0.0)
    b.add_function("ix", "rate_of__S")
    b.add_function(table, "0")
    b.add_inline_table_function_spec(table, [-10.0, 0.0, 10.0], [-10.0, 0.0, 10.0], "ix", "linear")
    b.add_function("h", f"2*{call}")
    b.add_reaction([], [s], "functional", "fS")
    b.add_reaction([], [t], "functional", "h")
    return bngsim.Model(_core=b.build())


@pytest.mark.parametrize(
    "call",
    ["tfun_kf()", "tfun_kf ()", "tfun_kf\t()", "tfun_kf", "2tfun_kf()/2", "2.0tfun_kf()/2", "kf"],
)
def test_a_table_is_read_by_whatever_names_it(call):
    """A table over a rule on rateOf(S), read by `T' = 2*<the table>`: T
    waits on rateOf(S) and takes a probe more. The expressions that read a
    table were found by `tfun_kf(` in their text; ExprTk compiles `tfun_kf
    ()`, a bare `tfun_kf` and `2tfun_kf()` as the same call, and with those
    the probe was not made: T(1) = 1.13 for 1.124 at the default tolerances
    and 15.2 at rtol 1e-6."""
    assert _table_on_a_rateof(call)._core.codegen_data()["rateof_flow_nesting"] == 1
    for kw in ({}, {"rtol": 1e-6, "atol": 1e-9}):
        sim = bngsim.Simulator(_table_on_a_rateof(call), method="ode")
        r = sim.run(sample_times=[0.0, 0.5, 1.0], **kw)
        assert _last(r, "T") == pytest.approx(2.0 * _S1, rel=2e-5)


@pytest.mark.parametrize("call", ["k.f", "tfun_k.f()", "2k.f/2"])
def test_a_table_whose_name_has_a_dot_in_it(call):
    """`k.f` is one name to ExprTk. Read as `k` and `f`, the call of a table
    so named was found in no expression (T(1) = 4.50 for 0.865 at rtol
    1e-6)."""
    assert _table_on_a_rateof(call, "k.f")._core.codegen_data()["rateof_flow_nesting"] == 1
    sim = bngsim.Simulator(_table_on_a_rateof(call, "k.f"), method="ode")
    r = sim.run(sample_times=[0.0, 0.5, 1.0], rtol=1e-6, atol=1e-9)
    assert _last(r, "T") == pytest.approx(2.0 * _S1, rel=2e-5)
