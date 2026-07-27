"""GH #65 — ``bngsim_dfdp`` must be able to read ``obs[]`` and ``func[]``.

``bngsim_codegen_sens_rhs`` is two halves, ``bngsim_dfdp`` (∂f/∂p_iP) and
``bngsim_jac_vec`` (J·yS). ``bngsim_dfdp`` was declared

    static void bngsim_dfdp(int iP, double t, const double* y,
                            const double* p, double* dfdp_out)

with no observable or function context, while every Functional rate-law
derivative is written in exactly those symbols — so the Functional ∂f/∂p of #66
had nowhere to land.

This stage is plumbing only, and its whole claim is that **Elementary emission
does not move**: an Elementary derivative ``∂(k·sf·∏y^m)/∂k`` is written purely
in ``p[]``/``y[]``, so the switch never mentions ``obs[``/``func[``, the value
thunk is never invoked, and the emitted source is byte-for-byte what it was.
The tests below pin both halves: the Elementary text (and that the thunk stays
untouched), and that the extended form actually works — compiled and called
through ctypes, flat and chunked, rather than merely emitted.

The vehicle for "a derivative that reads obs[]/func[]" is a ``derived_terms``
entry whose C expression references them. That is not a contrivance: #66's
Functional term has the identical shape —
``∂f_i/∂p = Σ_r stat_r·netstoich_ir·(∂func_r/∂p)·∏R_r`` is the same
"(derivative expression) × geometry" product the derived-rate-constant chain
rule already emits, with ``∂func_r/∂p`` in place of ``∂p_d/∂primary``.
"""

from __future__ import annotations

import ctypes
import subprocess

import pytest
from bngsim import _codegen as cg

# ─── harness (mirrors test_codegen_chunking.py) ────────────────────────────


class _SensUserData(ctypes.Structure):
    _fields_ = [
        ("param_values", ctypes.POINTER(ctypes.c_double)),
        ("plist", ctypes.POINTER(ctypes.c_int)),
        ("n_sens", ctypes.c_int),
    ]


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")

N_SP = 2
N_PAR = 3


def _rxn(derived_terms):
    """One reaction ``S0 -> S1`` with rate ``p[0]·y[0]``."""
    return [
        {
            "param_idx": 0,
            "stat_factor": 1.0,
            "stoich": {0: -1, 1: 1},
            "reactant_mult": {0: 1},
            "derived_terms": derived_terms,
        }
    ]


def _values():
    """obs[0] = y0 + y1 ; func[0] = p2 · obs[0] — both state- and
    parameter-coupled, so a wrong argument order shows up as a wrong number."""
    return (
        ["    double obs[1];", "    obs[0] = y[0] + y[1];"],
        ["    double func[1];", "    func[0] = p[2] * obs[0];"],
    )


def _signature(src: str) -> str:
    i = src.index("static void bngsim_dfdp")
    return src[i : src.index("{\n", i) + 2]


def _build_so(c_source: str, tmp_path, tag: str) -> str:
    c_path = tmp_path / f"{tag}.c"
    so_path = tmp_path / f"{tag}{cg._shared_lib_suffix()}"
    c_path.write_text(c_source)
    cmd = cg._build_compile_cmd(c_path, so_path, "-O1")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, f"compile failed for {tag}:\n{res.stderr}"
    return str(so_path)


def _call_sens(so_path, y, p, iP, yS=None, t=0.41):
    """``bngsim_codegen_sens_rhs`` for parameter ``iP``. An all-zero ``yS``
    zeroes J·yS exactly, so ySdot comes back as the bare ∂f/∂p column — the
    same trick steady_state.cpp's eval_dfdp uses."""
    lib = ctypes.CDLL(so_path)
    fn = lib.bngsim_codegen_sens_rhs
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    parr = (ctypes.c_double * N_PAR)(*p)
    pl = (ctypes.c_int * 1)(iP)
    ud = _SensUserData(param_values=parr, plist=pl, n_sens=1)
    ya = (ctypes.c_double * N_SP)(*y)
    ySa = (ctypes.c_double * N_SP)(*(yS or [0.0] * N_SP))
    ySdot = (ctypes.c_double * N_SP)()
    scratch = [(ctypes.c_double * N_SP)() for _ in range(3)]
    assert fn(1, t, ya, scratch[0], 0, ySa, ySdot, ctypes.byref(ud), *scratch[1:]) == 0
    return list(ySdot)


# ─── the no-behavior-change half ───────────────────────────────────────────


class TestElementaryUnchanged:
    def test_signature_is_the_pre_65_two_line_form(self):
        """The exact text, not just "compiles" — this is the stage's claim."""
        src = cg._emit_sens_rhs_body(_rxn([]), N_SP, N_PAR, set())
        assert _signature(src) == (
            "static void bngsim_dfdp(int iP, double t, const double* y,\n"
            "                        const double* p, double* dfdp_out) {\n"
        )
        assert "    bngsim_dfdp(iP, t, y, p, dfdp);" in src
        assert "obs[" not in src and "func[" not in src

    def test_value_thunk_is_never_invoked(self):
        """Not merely "the output is the same" — an Elementary model must not
        pay the expression translation at all, so the thunk must not run."""

        def _boom():
            raise AssertionError("value_lines_fn called for an Elementary model")

        src = cg._emit_sens_rhs_body(_rxn([]), N_SP, N_PAR, set(), value_lines_fn=_boom)
        assert "bngsim_dfdp(iP, t, y, p, dfdp)" in src

    def test_supplying_a_thunk_changes_nothing(self):
        assert cg._emit_sens_rhs_body(
            _rxn([]), N_SP, N_PAR, set(), value_lines_fn=_values
        ) == cg._emit_sens_rhs_body(_rxn([]), N_SP, N_PAR, set())

    def test_derived_rate_constant_chain_still_takes_the_short_form(self):
        """A #15 derived-parameter term is a p[]-only expression, so it must not
        trip the new context detection."""
        src = cg._emit_sens_rhs_body(
            _rxn([(1, "p[2]")]), N_SP, N_PAR, set(), value_lines_fn=_values
        )
        assert "const double* p, double* dfdp_out) {" in _signature(src)

    def test_elementary_model_with_observables_and_functions_stays_short(self, tmp_path):
        """The corpus case: an all-Elementary model that *has* obs/func still
        emits no obs/func into the sensitivity RHS."""
        import bngsim

        net = tmp_path / "m.net"
        net.write_text(
            "begin parameters\n 1 k 0.3\nend parameters\n"
            "begin species\n 1 A() 10.0\n 2 B() 0.0\nend species\n"
            "begin reactions\n 1 1 2 k\nend reactions\n"
            "begin observables\n 1 Molecules Atot 1\nend observables\n"
            "begin functions\n 1 fA()=Atot*2\nend functions\n"
        )
        src = cg.generate_sens_from_model(bngsim.Model.from_net(net))
        assert src is not None
        assert "const double* p, double* dfdp_out) {" in _signature(src)
        assert "double obs[" not in src and "double func[" not in src


# ─── the plumbing actually works half ──────────────────────────────────────


class TestExtendedContext:
    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            (
                "obs[0]",
                "const double* p, const double* obs,\n                        double* dfdp_out",
            ),
            (
                "func[0]",
                "const double* p, const double* func,\n                        double* dfdp_out",
            ),
            (
                "obs[0] * func[0]",
                "const double* p, const double* obs,\n"
                "                        const double* func, double* dfdp_out",
            ),
        ],
    )
    def test_obs_and_func_are_added_independently(self, expr, expected):
        """Mirrors how the analytical Jacobian picks its shard-block signature:
        a derivative that reads only one does not carry the other."""
        src = cg._emit_sens_rhs_body(_rxn([(1, expr)]), N_SP, N_PAR, set(), value_lines_fn=_values)
        assert expected in _signature(src)

    def test_func_only_still_computes_obs_it_depends_on(self):
        """func[0] is written in obs[0]; the driver must fill obs[] even though
        bngsim_dfdp does not receive it."""
        src = cg._emit_sens_rhs_body(
            _rxn([(1, "func[0]")]), N_SP, N_PAR, set(), value_lines_fn=_values
        )
        assert "    obs[0] = y[0] + y[1];" in src
        assert "bngsim_dfdp(iP, t, y, p, func, dfdp);" in src

    def test_declines_when_no_context_is_available(self):
        """The .net path passes no thunk. Refuse rather than emit C that names
        an undeclared obs[] — the #56 "decline loudly" precedent."""
        assert cg._emit_sens_rhs_body(_rxn([(1, "obs[0]")]), N_SP, N_PAR, set()) is None

    def test_declines_when_the_thunk_declines(self):
        assert (
            cg._emit_sens_rhs_body(
                _rxn([(1, "obs[0]")]), N_SP, N_PAR, set(), value_lines_fn=lambda: None
            )
            is None
        )

    @requires_cc
    def test_compiles_and_the_values_arrive(self, tmp_path):
        """End-to-end: the arrays reach bngsim_dfdp with the right contents.

        ∂f/∂p_1 = (obs0·func0)·y0 scattered as (-1, +1), with
        obs0 = y0+y1 and func0 = p2·obs0.
        """
        src = cg._emit_sens_rhs_body(
            _rxn([(1, "obs[0] * func[0]")]), N_SP, N_PAR, set(), value_lines_fn=_values
        )
        so = _build_so(src, tmp_path, "sens_ctx")
        y, p = [2.0, 3.0], [0.7, 1.0, 5.0]

        obs0 = y[0] + y[1]
        func0 = p[2] * obs0
        v = obs0 * func0 * y[0]
        assert _call_sens(so, y, p, iP=1) == pytest.approx([-v, v])

        # The Elementary column through the same binary is untouched by the
        # added context: ∂f/∂p_0 = y0.
        assert _call_sens(so, y, p, iP=0) == pytest.approx([-y[0], y[0]])

        # A parameter that is no reaction's rate constant still hits `default:`
        # and returns zero — the sentinel IC-sensitivity path (cvode_simulator).
        assert _call_sens(so, y, p, iP=N_PAR) == pytest.approx([0.0, 0.0])

    @requires_cc
    def test_jac_vec_term_is_still_added(self, tmp_path):
        """A non-zero yS must still contribute J·yS on top of ∂f/∂p."""
        src = cg._emit_sens_rhs_body(
            _rxn([(1, "obs[0] * func[0]")]), N_SP, N_PAR, set(), value_lines_fn=_values
        )
        so = _build_so(src, tmp_path, "sens_ctx_jv")
        y, p, yS = [2.0, 3.0], [0.7, 1.0, 5.0], [1.5, 0.0]
        obs0 = y[0] + y[1]
        v = obs0 * (p[2] * obs0) * y[0]
        jv = p[0] * yS[0]  # dv/dy0 = k
        assert _call_sens(so, y, p, iP=1, yS=yS) == pytest.approx([-v - jv, v + jv])


class TestChunkedContext:
    """Above the chunk threshold the obs[]/func[] fill is lifted into NOINLINE
    shard blocks (GH #165) that compile_rhs splits into parallel translation
    units. The sensitivity blocks take no ``user_data`` — the sens RHS holds a
    CodegenSensUserData, which has no tfun callback — so this needs its own
    signature and its own coverage."""

    @staticmethod
    def _wide(n_rxn):
        return [
            {
                "param_idx": 0,
                "stat_factor": 1.0,
                "stoich": {0: -1, 1: 1},
                "reactant_mult": {0: 1},
                "derived_terms": [(1, "obs[0] * func[0]")] if i == 0 else [],
            }
            for i in range(n_rxn)
        ]

    def test_shard_blocks_are_emitted_and_take_no_user_data(self, monkeypatch):
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "4")
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", "2")
        src = cg._emit_sens_rhs_body(self._wide(8), N_SP, N_PAR, set(), value_lines_fn=_values)
        assert src is not None
        assert "sens_obs_blk_000" in src and "sens_func_blk_000" in src
        assert "user_data" not in src[: src.index("typedef struct {")]
        # Lifted on the same sentinels compile_rhs splits the RHS on.
        assert src.count(cg._SHARD_BLOCK_OPEN) == src.count(cg._SHARD_BLOCK_CLOSE)

    @requires_cc
    def test_chunked_result_matches_flat(self, tmp_path, monkeypatch):
        flat = cg._emit_sens_rhs_body(self._wide(8), N_SP, N_PAR, set(), value_lines_fn=_values)
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "4")
        monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK_SIZE", "2")
        chunked = cg._emit_sens_rhs_body(self._wide(8), N_SP, N_PAR, set(), value_lines_fn=_values)
        assert flat != chunked
        y, p = [2.0, 3.0], [0.7, 1.0, 5.0]
        for iP in (0, 1, N_PAR):
            assert _call_sens(
                _build_so(flat, tmp_path, f"flat{iP}"), y, p, iP=iP
            ) == pytest.approx(_call_sens(_build_so(chunked, tmp_path, f"chunk{iP}"), y, p, iP=iP))


# ─── the model-side context builder ────────────────────────────────────────


def _data(functions, tfuns=()):
    return {
        "parameters": [{"name": "k"}, {"name": "kd"}, {"name": "s"}],
        "species": [{"name": "A"}, {"name": "B"}],
        "observables": [{"name": "Atot", "entries": [(0, 1.0)]}],
        "functions": [{"name": n, "expression": e} for n, e in functions],
        "table_functions": list(tfuns),
    }


class TestSensValueLines:
    def test_emits_the_shared_value_lines(self):
        obs, func = cg._sens_value_lines(_data([("fA", "Atot*2")]))
        assert obs == ["    double obs[1];", "    obs[0] = y[0];  /* Atot */"]
        assert func == ["    double func[1];", "    func[0] = obs[0]*2.0;  /* fA */"]

    def test_declines_a_whole_body_table_function(self):
        """_emit_function_lines writes these as data->tfun_eval(...), and the
        sens RHS holds a CodegenSensUserData with no tfun_ctx/tfun_eval."""
        data = _data([("drive", "tfun()")], tfuns=[{"name": "drive", "index_kind": "time"}])
        assert cg._sens_value_lines(data) is None

    def test_declines_an_embedded_tfun_wrapper(self):
        data = _data([("fA", "tfun_drive__tfun0() + 5")], tfuns=[{"name": "drive__tfun0"}])
        assert cg._sens_value_lines(data) is None

    def test_declines_rateof(self):
        assert cg._sens_value_lines(_data([("fA", f"{cg._RATEOF_PREFIX}A * 2")])) is None
