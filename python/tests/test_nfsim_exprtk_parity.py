"""Behavioural tests for NFsim's ExprTk ``mu::Parser`` shim (issue #49).

The embedded NFsim path has two ExprTk-based expression layers: the host
``bngsim::ExprTkEvaluator`` (``bngsim/src/expression.cpp``) and NFsim's internal
``mu::Parser`` shim (``third_party/nfsim/src/NFfunction/nfsim_funcparser.h``).
The shim needs these BNG-compatibility features:

  * the ``mratio(a, b, z)`` built-in -- the confluent-hypergeometric ratio
    ``M(a+1, b+1, z) / M(a, b, z)``; and
  * a reserved-symbol remap, so a model symbol named like an ExprTk built-in
    (e.g. ``frac``) resolves to the model quantity instead of the built-in.
  * the ``rint(x)`` built-in, BNG's ``floor(x + 0.5)`` (issue #771); and
  * numeric literals read as strtod reads them, not as ExprTk's own parser does.

Per issue #49 the shim no longer carries a hand-ported copy of this logic: it
**forwards** each feature to the host single source ``bngsim::expr_compat``
(``mratio``, ``rint`` and ``compute_registration_name`` / ``remap_name``), linked into the
vendored NFsim target via ``bngsim::expression``. There is therefore no second
implementation to drift -- so the old host-vs-shim parity assertion has been
retired as tautological.

What these tests still pin, and why it remains meaningful: the function columns
read below are evaluated *inside NFsim* via ``FuncFactory::Eval`` (see
``bngsim/src/nfsim_simulator.cpp`` ``eval_output_function``) -- i.e. through the
shim's ``mu::Parser`` adapters, which forward to the host. Anchoring those
columns to an *independent* oracle (``scipy.special.hyp1f1`` for ``mratio``,
arithmetic for the reserved remap) confirms end-to-end that the adapter forwards
correctly *and* that the vendored NFsim object actually links the host's single
source at runtime -- not merely that the code compiles.

The ``m_*`` functions take only constant arguments, so their value depends solely
on the ``mratio`` implementation, not on the (stochastic) trajectory; the
``frac`` observable counts molecules of ``F``, which no rule touches, so it is a
fixed ``7`` for all time. See ``tests/data/nfsim/exprtk_shim_parity.xml``.
"""

from __future__ import annotations

import numpy as np
import pytest

# scipy supplies the independent oracle for mratio (Kummer's M == hyp1f1).
scipy_special = pytest.importorskip("scipy.special")

from bngsim._bngsim_core import NfsimSimulator, TimeSpec  # noqa: E402

# (a, b, z) triples baked into exprtk_shim_parity.xml as the m_* function
# columns, chosen to span regimes the modified-Lentz continued fraction in
# mratio must handle: moderate z, a<b with larger z, negative z, and large z
# with b<1.
_MRATIO_CASES = {
    "m_moderate": (2.5, 1.3, 0.7),
    "m_alt": (0.5, 2.0, 3.0),
    "m_negz": (4.0, 1.5, -1.2),
    "m_bigz": (3.2, 0.8, 5.0),
}

# The fixed value of the `frac` observable (count of F) in the fixture.
_FRAC_OBS = 7.0


def _mratio_reference(a: float, b: float, z: float) -> float:
    """mratio(a,b,z) = M(a+1,b+1,z) / M(a,b,z), where M is Kummer's confluent
    hypergeometric function == ``scipy.special.hyp1f1``. Independent of bngsim's
    continued-fraction port, so it is a true oracle rather than a second copy."""
    return scipy_special.hyp1f1(a + 1.0, b + 1.0, z) / scipy_special.hyp1f1(a, b, z)


@pytest.fixture
def parity_sim_cols(data_dir):
    """Run the parity fixture under NFsim and return (sim, {col_name: series}).

    ``series`` is the column's value at every output time; the shim recomputes
    each function at each output, so these series let later tests assert
    time-invariance as well as value.
    """
    xml = data_dir / "nfsim" / "exprtk_shim_parity.xml"
    sim = NfsimSimulator(str(xml))
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 5.0
    ts.n_points = 4
    res = sim.run(ts, 12345)
    names = list(res.expression_names)
    table = np.array([np.asarray(res.expression_data[t]).ravel() for t in range(res.n_times)])
    cols = {name: table[:, i] for i, name in enumerate(names)}
    return sim, cols


@pytest.mark.parametrize("col, abz", list(_MRATIO_CASES.items()))
def test_nfsim_shim_mratio_matches_scipy(parity_sim_cols, col, abz):
    """NFsim shim's ``mratio(a,b,z)`` equals scipy's confluent-hypergeometric
    ratio. Oracle: ``scipy.special.hyp1f1`` (independent implementation).

    rtol=1e-10 is conservative: the host/scipy agreement observed for these
    triples is ~1e-16, while a genuinely broken port (wrong recurrence, dropped
    term, sign error) would be off by a relative amount far above 1e-10.
    """
    _, cols = parity_sim_cols
    assert col in cols, f"{col} did not compile/appear as an NFsim function column"
    np.testing.assert_allclose(cols[col], _mratio_reference(*abz), rtol=1e-10)


# rint(x) columns baked into exprtk_shim_parity.xml, with BNG's value for each:
# floor(x + 0.5), the definition BNG2.pl's run_network evaluates (issue #771).
# The negative halves are the discriminating cases: std::round, which the shim
# used, gives -3, -2 and -1 for the first three. (The 0.49999999999999994 edge is
# pinned in test_rint_bng_semantics.py through a parameter value.)
_RINT_CASES = {
    "rint_m2p5": -2.0,
    "rint_m1p5": -1.0,
    "rint_m0p5": 0.0,
    "rint_p0p5": 1.0,
    "rint_p2p5": 3.0,
    "rint_m0p3": 0.0,
}


@pytest.mark.parametrize("col, want", list(_RINT_CASES.items()))
def test_nfsim_shim_rint_is_bng_floor_half(parity_sim_cols, col, want):
    """NFsim shim's ``rint(x)`` is BNG's ``floor(x + 0.5)``, evaluated inside
    NFsim through the forwarding adapter (issue #771). Oracle: the literal
    values of floor(x + 0.5), written out above."""
    _, cols = parity_sim_cols
    assert col in cols, f"{col} did not compile/appear as an NFsim function column"
    np.testing.assert_array_equal(cols[col], want)


# Literal columns in exprtk_shim_parity.xml. ExprTk's own number parser read
# 4.82e-06 and 2.71828182845904523536 one ulp off and 0.49999999999999997 as 0.5
# (issue #1013);
# the shim now passes its expression through the host's exact_literals, so each
# reads as strtod (Python's float()) reads it.
_LITERAL_CASES = {
    "lit_sci": float("4.82e-06"),
    "lit_long": float("2.71828182845904523536"),
    "lit_below_half": 1.0,  # 0.49999999999999997 < 0.5 only when read right
}


@pytest.mark.parametrize("col, want", list(_LITERAL_CASES.items()))
def test_nfsim_shim_reads_a_literal_as_strtod(parity_sim_cols, col, want):
    _, cols = parity_sim_cols
    assert col in cols, f"{col} did not compile/appear as an NFsim function column"
    np.testing.assert_array_equal(cols[col], want)


def test_nfsim_shim_resolves_reserved_symbol(parity_sim_cols):
    """A model observable named ``frac`` (an ExprTk-reserved built-in) used as a
    bare value resolves to the observable, so ``frac + 10`` == 17.

    Oracle: arithmetic -- the ``frac`` observable is a fixed 7, so 7 + 10 = 17.
    This is discriminating: ExprTk's built-in ``frac`` is a unary function, so a
    bare ``frac`` only compiles if the shim remaps the colliding model symbol to
    ``r_frac``. Without the carry the function fails to compile (ERR029) and this
    column would be absent (KeyError below) or the run would raise.
    """
    _, cols = parity_sim_cols
    assert "reserved_frac" in cols, (
        "reserved_frac did not compile -- the shim's reserved-symbol remap is "
        "missing, so bare `frac` hit ExprTk's built-in"
    )
    np.testing.assert_allclose(cols["reserved_frac"], _FRAC_OBS + 10.0, atol=1e-12)


def test_nfsim_shim_parity_columns_are_time_invariant(parity_sim_cols):
    """Invariant: every parity column depends only on constants (m_*) or a fixed
    observable (reserved_frac), so each is constant across output times. Confirms
    the compared values come from the deterministic expression layer rather than
    a stochastic trajectory column. atol=1e-12 still flags an O(1) drift from a
    truly state-dependent value.
    """
    _, cols = parity_sim_cols
    for name, series in cols.items():
        np.testing.assert_allclose(
            series, series[0], atol=1e-12, err_msg=f"{name} varied across output times"
        )
