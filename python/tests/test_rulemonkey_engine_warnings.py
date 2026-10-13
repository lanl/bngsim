"""GH #1022: what RuleMonkey flags in an XML without refusing it reaches Python.

RuleMonkey reports each thing it cannot honor as an error, which bngsim turns
into a refusal, or as a warning, which bngsim used to drop. The warnings are now
raised as :class:`bngsim.RuleMonkeyWarning` when a ``method="rm"`` Simulator or
a RuleMonkeySession is built.

The case that made this matter is a rule whose products can never be what it
writes (richardposner/RuleMonkey#105). ``A(s~0) + B(a) -> A(s~1).B(a)`` writes A
and B into one complex but bonds nothing. BioNetGen generates no reaction for
it. RuleMonkey 3.10.1 fired it, at the full bimolecular rate; 3.11.0 never
fires it and says so.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import RuleMonkeyWarning


def _has_rulemonkey() -> bool:
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_rulemonkey(),
    reason="RuleMonkey not compiled in",
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"
# never_join: A(s~0) + B(a) -> A(s~1).B(a) k, and control: C(s~0) -> C(s~1) k,
# k = 1, 1000 of each. The .net is BNG2.pl's network of the same model: it has
# the control reaction only.
NEVER = DATA / "products_never_form.xml"
NEVER_NET = DATA / "products_never_form.net"
PLAIN = DATA / "first_order_switch.xml"


def _model():
    return bngsim.Model.from_net(str(NEVER_NET))


def test_a_rule_whose_products_never_form_never_fires():
    """A1 stays at 0, as on the generated network. On 3.10.1 it read 1000 by
    t = 1. The control rule runs as it should: C1 = 1000*(1 - exp(-t))."""
    with pytest.warns(RuleMonkeyWarning):
        sim = bngsim.Simulator(_model(), method="rm", xml_path=str(NEVER))
    r = sim.run(t_span=(0.0, 2.0), n_points=5, seed=1)
    t = np.asarray(r.time)
    np.testing.assert_array_equal(np.asarray(r.observables["A1"]), 0.0)
    p = 1.0 - np.exp(-t)
    c1 = np.asarray(r.observables["C1"])
    np.testing.assert_array_less(
        np.abs(c1 - 1000.0 * p), 5.0 * np.sqrt(1000.0 * p * (1 - p)) + 1e-9
    )

    ode = bngsim.Simulator(_model(), method="ode").run(t_span=(0.0, 2.0), n_points=5)
    np.testing.assert_array_equal(np.asarray(ode.observables["A1"]), 0.0)
    np.testing.assert_allclose(np.asarray(ode.observables["C1"]), 1000.0 * p, rtol=1e-6, atol=1e-6)


def test_the_simulator_says_which_rule_and_why():
    with pytest.warns(RuleMonkeyWarning) as caught:
        bngsim.Simulator(_model(), method="rm", xml_path=str(NEVER))
    assert len(caught) == 1
    message = str(caught[0].message)
    assert message.startswith("RuleMonkey: ReactionRule: Rule 'RR1' (never_join)")
    assert "RM never fires it" in message
    # Attributed to the line that built the simulator, not to bngsim's own code.
    assert caught[0].filename == __file__


def test_the_session_says_it_too():
    with pytest.warns(RuleMonkeyWarning, match=r"\(never_join\).*never fires") as caught:
        bngsim.RuleMonkeySession(str(NEVER)).destroy()
    assert len(caught) == 1
    assert caught[0].filename == __file__


def test_a_model_the_engine_does_not_flag_raises_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bngsim.Simulator(_model(), method="rm", xml_path=str(PLAIN))
        bngsim.RuleMonkeySession(str(PLAIN)).destroy()


def test_an_older_flag_is_surfaced_as_well():
    """TotalRate has been flagged by the engine all along: BioNetGen's network
    does not implement it, so the model has no network result to check against."""
    with pytest.warns(RuleMonkeyWarning, match="TotalRate"):
        bngsim.RuleMonkeySession(str(DATA / "symmetry_factor_total_rate.xml")).destroy()


def test_the_warning_can_be_escalated():
    assert issubclass(RuleMonkeyWarning, UserWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuleMonkeyWarning)
        with pytest.raises(RuleMonkeyWarning):
            bngsim.RuleMonkeySession(str(NEVER))


def test_the_core_lists_what_the_engine_flagged():
    from bngsim._bngsim_core import RuleMonkeySimulator

    assert RuleMonkeySimulator(str(PLAIN)).warnings() == []
    flagged = RuleMonkeySimulator(str(NEVER)).warnings()
    assert len(flagged) == 1 and flagged[0].startswith("ReactionRule: Rule 'RR1' (never_join)")
