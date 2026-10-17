"""A post-init parameter write reaches every NFsim rate law; NFsim refuses what it cannot run.

Issue #854's review found three more ways the vendored NFsim ran a rule at a rate
its parameters no longer gave, with no warning:

* A Michaelis-Menten rule held kcat and Km as numbers with no parameter name, so
  ``NfsimSession.set_param('kcat', ...)`` after ``initialize()`` changed
  ``get_parameter('kcat')`` and left the rule at its old kcat (NFsim carry 0018).
* NFinput folded a zero-order rule's volume conversion in through ``setBaseRate()``,
  which applies the symmetry factor a second time. The rule loaded at sf**2 * kp,
  and the first parameter write of any kind, even a no-op, moved it to sf * kp
  (carry 0017). BNG writes ``symmetry_factor="1"`` for every zero-order rule, so
  only XML written some other way reaches this.
* Arrhenius (energy) rules: NFsim's expansion divides every energy by an RT that
  BioNetGen does not apply and gets one-way rules wrong (RuleWorld/nfsim#102).
  They are refused by name until the vendored NFsim carries the upstream fix,
  and so is a rate-law type NFsim does not implement: it dropped a Hill rule.
* A function NFsim cannot build -- a local-function rate law reading both
  reactants, a local function reading a Species observable or taking two
  arguments, a function calling a function and reading an observable -- made
  NFsim call exit(1) and kill the Python process (carry 0019). The load now
  fails with an error.

The oracle for a post-init write is the same model with the new value written
into the XML and no override at all. That load path never runs the refresh, so
it cannot share a defect with it, and with the same seed both runs make the same
draws: the trajectories must be identical. Each case also checks a closed form.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import ModelError
from bngsim._network_free_checks import arrhenius_rules, nfsim_unimplemented_rate_laws

pytestmark = pytest.mark.skipif(
    not getattr(bngsim, "HAS_NFSIM", False),
    reason="NFsim not compiled in",
)


def _write(src: Path, dst_dir: Path, params: dict[str, float], swap=()) -> Path:
    """A copy of *src* with each named ``<Parameter>``'s value written in, and each
    ``(old, new)`` pair of *swap* replaced exactly once."""
    text = src.read_text()
    for name, value in params.items():
        text, n = re.subn(
            rf'(<Parameter id="{name}" type="Constant" value=")[^"]*(" expr=")[^"]*(")',
            rf"\g<1>{value!r}\g<2>{value!r}\g<3>",
            text,
        )
        assert n == 1, name
    for old, new in swap:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    out = dst_dir / src.name
    out.write_text(text)
    return out


def _run(
    xml: Path, *, pre=None, post=None, seed=1, t_end=4.0, n_points=5, core=False
) -> np.ndarray:
    """Observables at n_points times. With *core*, through the core
    NfsimSimulator, which the NfsimSession refusals do not guard: NfsimSession
    refuses a compartment model (GH #1067), and this still pins the vendored
    zero-order volume refresh."""
    if core:
        from bngsim._bngsim_core import NfsimSimulator

        sim = NfsimSimulator(str(xml))
        for name, value in (pre or {}).items():
            sim.set_param(name, value)
        sim.initialize(seed)
        for name, value in (post or {}).items():
            sim.set_param(name, value)
        out = np.asarray(sim.simulate(0, t_end, n_points).observable_data).ravel()
        sim.destroy_session()
        return out
    with bngsim.NfsimSession(str(xml)) as nf:
        for name, value in (pre or {}).items():
            nf.set_param(name, value)
        nf.initialize(seed=seed)
        for name, value in (post or {}).items():
            nf.set_param(name, value)
        return np.asarray(nf.simulate(0, t_end, n_points=n_points).observables).ravel()


# ─── Michaelis-Menten: kcat and Km follow a post-init write ─────────────────


@pytest.fixture
def mm_xml(data_dir: Path) -> Path:
    """``S() + E() -> P() + E() MM(kcat, Km)``, 1000 S and 10 E, kcat = 0 at load."""
    return data_dir / "nfsim" / "mm_switchable_rate.xml"


def _mm_mean_product(t: np.ndarray, kcat=1.0, km=100.0, e=10.0, s0=1000.0) -> np.ndarray:
    """dP/dt = kcat*E*S/(Km+S), S = s0 - P, by RK4: the textbook law, which is
    within 0.3 % of the quasi-steady form NFsim and BNG use at E/S0 = 1 %."""
    rate = lambda p: kcat * e * (s0 - p) / (km + s0 - p)  # noqa: E731
    out, p, h, now = [], 0.0, 1e-3, 0.0
    for target in t:
        while now < target - 1e-12:
            k1 = rate(p)
            k2 = rate(p + h * k1 / 2)
            k3 = rate(p + h * k2 / 2)
            k4 = rate(p + h * k3)
            p += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            now += h
        out.append(p)
    return np.array(out)


def test_a_post_init_kcat_runs_as_kcat_written_into_the_xml(mm_xml, tmp_path):
    post = _run(mm_xml, post={"kcat": 1.0})
    written = _run(_write(mm_xml, tmp_path, {"kcat": 1.0}))
    np.testing.assert_array_equal(post, written)
    assert post[-1] > 0  # before the fix it stayed at 0: the rule kept kcat = 0


def test_a_post_init_kcat_gives_the_michaelis_menten_rate(mm_xml):
    t = np.linspace(0, 4, 5)
    runs = np.array([_run(mm_xml, post={"kcat": 1.0}, seed=s) for s in range(1, 11)])
    expected = _mm_mean_product(t)
    # P(t) is close to Poisson: its spread is sqrt(mean). Five standard errors.
    tol = 5 * np.sqrt(np.maximum(expected, 1.0) / len(runs))
    np.testing.assert_array_less(np.abs(runs.mean(axis=0) - expected), tol)


def test_a_post_init_km_runs_as_km_written_into_the_xml(mm_xml, tmp_path):
    post = _run(_write(mm_xml, tmp_path, {"kcat": 1.0}), post={"Km": 10.0})
    (tmp_path / "km").mkdir()
    written = _run(_write(mm_xml, tmp_path / "km", {"kcat": 1.0, "Km": 10.0}))
    np.testing.assert_array_equal(post, written)


def test_a_post_init_kcat_keeps_the_rule_symmetry_factor(mm_xml, tmp_path):
    # baseRate carries an MM rule's symmetry factor; the refresh must leave it be.
    sym = _write(mm_xml, tmp_path, {}, swap=[('symmetry_factor="1"', 'symmetry_factor="0.5"')])
    post = _run(sym, post={"kcat": 1.0})
    (tmp_path / "w").mkdir()
    written = _run(_write(sym, tmp_path / "w", {"kcat": 1.0}))
    np.testing.assert_array_equal(post, written)
    assert post[-1] > 0


def test_a_post_init_write_reaches_a_derived_kcat(mm_xml, tmp_path):
    # kcat2 = 2*kcat, as BNG writes a derived parameter; the rule reads kcat2.
    derived = _write(
        mm_xml,
        tmp_path,
        {},
        swap=[
            (
                '<Parameter id="Km" type="Constant" value="100" expr="100"/>',
                '<Parameter id="Km" type="Constant" value="100" expr="100"/>\n'
                '      <Parameter id="kcat2" type="Constant" value="0" expr="2*kcat"/>',
            ),
            ('<RateConstant value="kcat"/>', '<RateConstant value="kcat2"/>'),
        ],
    )
    post = _run(derived, post={"kcat": 0.5})
    (tmp_path / "w").mkdir()
    written = _run(_write(derived, tmp_path / "w", {"kcat": 0.5, "kcat2": 1.0}))
    np.testing.assert_array_equal(post, written)
    assert post[-1] > 0


def test_a_post_init_kcat_of_zero_stops_the_rule(mm_xml, tmp_path):
    out = _run(_write(mm_xml, tmp_path, {"kcat": 1.0}), post={"kcat": 0.0})
    np.testing.assert_array_equal(out, 0.0)


# ─── Zero-order: the volume fold applies the symmetry factor once ───────────


def _birth_death(t: np.ndarray, kp: float, kdeg=0.7, a0=5000.0):
    """Mean and variance of A(t) for 0 -> A at kp, A -> 0 at kdeg, A(0) = a0:
    binomial survivors of a0 plus Poisson immigrants."""
    p = np.exp(-kdeg * t)
    mean = a0 * p + kp / kdeg * (1 - p)
    var = a0 * p * (1 - p) + kp / kdeg * (1 - p)
    return mean, var


@pytest.fixture
def symmetric_zero_order_xml(nfsim_zero_order_switchable_rate_xml, tmp_path) -> Path:
    """The GH #854 birth-death model with kp = 1000 written in and the synthesis
    rule carrying symmetry_factor="0.5", which BNG never writes for a zero-order
    rule but the XML format allows."""
    return _write(
        nfsim_zero_order_switchable_rate_xml,
        tmp_path,
        {"kp": 1000.0},
        swap=[('name="_R2" symmetry_factor="1"', 'name="_R2" symmetry_factor="0.5"')],
    )


def test_a_parameter_write_leaves_a_zero_order_rate_where_it_loaded(symmetric_zero_order_xml):
    loaded = _run(symmetric_zero_order_xml)
    # A write of another parameter, at the value it already has.
    after_write = _run(symmetric_zero_order_xml, post={"kdeg": 0.7})
    np.testing.assert_array_equal(loaded, after_write)


def test_a_zero_order_rate_carries_its_symmetry_factor_once(symmetric_zero_order_xml):
    t = np.linspace(0, 4, 5)[1:]
    runs = np.array([_run(symmetric_zero_order_xml, seed=s)[1:] for s in range(1, 9)])
    mean, var = _birth_death(t, kp=0.5 * 1000.0)
    tol = 5 * np.sqrt(var / len(runs))
    np.testing.assert_array_less(np.abs(runs.mean(axis=0) - mean), tol)
    # It loaded at 0.25 * kp before the fix: 639 at t = 4 against 975.
    wrong, _ = _birth_death(t, kp=0.25 * 1000.0)
    assert np.all(np.abs(runs.mean(axis=0) - wrong) > tol)


@pytest.mark.parametrize(
    "fixture, kp_v, core",
    [
        ("nfsim_zero_order_switchable_rate_xml", 1000.0, False),
        # A compartment model: NfsimSession refuses it (GH #1067), the core runs it.
        ("nfsim_zero_order_switchable_rate_compartment_xml", 2000.0, True),
    ],
)
def test_a_post_init_kp_runs_as_kp_written_into_the_xml(request, tmp_path, fixture, kp_v, core):
    xml = request.getfixturevalue(fixture)
    post = _run(xml, post={"kp": 1000.0}, core=core)
    written = _run(_write(xml, tmp_path, {"kp": 1000.0}), core=core)
    np.testing.assert_array_equal(post, written)
    # Synthesis at kp * V: V = 1 without a compartment, 2 in the compartment of size 2.
    mean, var = _birth_death(np.linspace(0, 4, 5), kp=kp_v)
    assert abs(post[-1] - mean[-1]) < 5 * np.sqrt(var[-1])


# ─── Arrhenius: refused on NFsim by name ─────────────────────────────────────


@pytest.fixture
def arrhenius_xml(data_dir: Path) -> Path:
    """Two reversible Arrhenius binding rules and a cooperativity energy pattern."""
    return data_dir / "nfsim" / "energy_arrhenius.xml"


_REFUSAL = (
    r"Arrhenius \(energy-based\) rate laws: "
    r"rules '_R1', '_reverse__R1', '_R2', '_reverse__R2'.*RuleWorld/nfsim#102"
)


def test_the_session_refuses_an_arrhenius_model(arrhenius_xml):
    with pytest.raises(ModelError, match=_REFUSAL):
        bngsim.NfsimSession(str(arrhenius_xml))


def test_the_simulator_refuses_an_arrhenius_model(arrhenius_xml, simple_decay_net):
    m = bngsim.Model.from_net(str(simple_decay_net))
    with pytest.raises(ModelError, match=_REFUSAL):
        bngsim.Simulator(m, method="nf", xml_path=str(arrhenius_xml))


def test_a_rate_law_nfsim_does_not_implement_is_refused(data_dir: Path, simple_decay_net):
    # BNG2.pl writes Hill into the XML; NFsim dropped the rule and P stayed at 0.
    xml = data_dir / "nfsim" / "hill_rate_law.xml"
    match = r"rate laws NFsim does not implement: rule '_R1' \(Hill\)"
    with pytest.raises(ModelError, match=match):
        bngsim.NfsimSession(str(xml))
    m = bngsim.Model.from_net(str(simple_decay_net))
    with pytest.raises(ModelError, match=match):
        bngsim.Simulator(m, method="nf", xml_path=str(xml))


@pytest.mark.parametrize(
    "fixture",
    [
        "mm_switchable_rate.xml",  # MM
        "zero_order_switchable_rate.xml",  # Ele
        "energy_patterns_only.xml",  # Ele, beside an energy pattern
    ],
)
def test_rate_laws_nfsim_implements_are_not_refused(data_dir: Path, fixture):
    xml = data_dir / "nfsim" / fixture
    assert arrhenius_rules(xml) == []
    assert nfsim_unimplemented_rate_laws(xml) == []
    bngsim.NfsimSession(str(xml)).destroy()


# ─── A rule NFsim cannot build fails the load instead of the process ─────────


@pytest.mark.parametrize(
    "fixture, says",
    [
        # fA(x)*fB(y): a local-function rate law reading both reactants.
        ("local_function_two_reactants.xml", "only support one DOR reactant"),
        # The same rule with fA and fB reading Species observables.
        ("local_function_species_observable.xml", "clone species observable"),
        # fAB(x,y): one local function with an argument for each reactant.
        ("local_function_two_arguments.xml", "maximum of 1 argument"),
        # g() = f() + k*SB: a function calling a function and reading an observable.
        ("composite_function_reads_observable.xml", "Composite Functions"),
    ],
)
def test_a_rule_nfsim_cannot_build_fails_the_load_not_the_process(data_dir: Path, fixture, says):
    # In a child process: before carry 0019, NFsim called exit(1) and took the
    # interpreter with it, which would end this test run, not fail one test.
    xml = data_dir / "nfsim" / fixture
    script = (
        "import sys, bngsim\n"
        "s = bngsim.NfsimSession(sys.argv[1])\n"
        "try:\n"
        "    s.initialize(seed=1)\n"
        "except bngsim.BngsimError as e:\n"
        "    print('RAISED', type(e).__name__, e)\n"
        "else:\n"
        "    print('RAN')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(xml)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.startswith("RAISED SimulationError"), proc.stdout
    assert says in proc.stdout, proc.stdout


def test_energy_patterns_without_an_arrhenius_rule_still_run(data_dir: Path):
    # A pattern changes no rate unless an Arrhenius rule reads it.
    xml = data_dir / "nfsim" / "energy_patterns_only.xml"
    assert arrhenius_rules(xml) == []
    out = _run(xml, t_end=10.0, n_points=3)
    assert out[0] == 0 and out[-1] > 0
