"""Tests for code-generated ODE RHS (Session 21).

Tests the full pipeline: .net parsing → C code generation → compilation
→ dlopen → CVODE integration → correctness vs ExprTk baseline.
"""

import os
import sys
import types

import numpy as np
import pytest
from bngsim._codegen import (
    _parse_net_file,
    _replace_power_op,
    compute_model_hash,
    generate_rhs_c,
    prepare_codegen,
)

# Honor BNGSIM_TEST_DATA so this module works under run_tests.sh, which copies
# tests to a temp dir (breaking __file__-relative resolution).
DATA = os.environ.get("BNGSIM_TEST_DATA") or os.path.join(
    os.path.dirname(__file__), "..", "..", "tests", "data"
)


class TestReplacePowerOp:
    """`^` → pow() translation must be whitespace-robust (GH #152).

    Spaced operators (``x ^ 2``) are natural in hand-authored functional rate
    laws. The ExprTk interpreter tolerates the spacing, but codegen used to drop
    the base operand when whitespace preceded ``^`` — emitting invalid C of the
    form ``pow(, exponent)`` that failed to compile.
    """

    @pytest.mark.parametrize(
        "tight,spaced",
        [
            ("x^2", "x ^ 2"),
            ("(a+b)^c", "(a+b) ^ c"),
            ("(a/b)^h", "(a/b) ^ h"),
            ("p[0]^h", "p[0] ^ h"),
            ("a^(b^c)", "a ^ (b ^ c)"),
        ],
    )
    def test_spaced_matches_tight(self, tight, spaced):
        out = _replace_power_op(spaced)
        # The base must survive — no `pow(,` artifact.
        assert "pow(," not in out
        # Spacing around `^` must not change the translation.
        assert out == _replace_power_op(tight)

    def test_simple_spaced(self):
        assert _replace_power_op("x ^ 2") == "pow(x, 2)"

    def test_parenthesized_spaced(self):
        assert _replace_power_op("(a+b) ^ c") == "pow((a+b), c)"

    def test_no_caret_is_passthrough(self):
        assert _replace_power_op("k0 * A_obs / 10") == "k0 * A_obs / 10"


class TestNetParser:
    """Test lightweight .net file parser."""

    def test_parse_simple_decay(self):
        model = _parse_net_file(os.path.join(DATA, "simple_decay.net"))
        assert len(model["parameters"]) == 1
        assert len(model["species"]) == 2
        assert len(model["reactions"]) == 1
        assert len(model["observables"]) == 2
        assert model["parameters"][0][1] == "k1"

    def test_parse_reversible(self):
        model = _parse_net_file(os.path.join(DATA, "two_species_reversible.net"))
        assert len(model["parameters"]) == 2
        assert len(model["species"]) == 3
        assert len(model["reactions"]) == 2

    def test_parse_observables(self):
        model = _parse_net_file(os.path.join(DATA, "simple_decay.net"))
        # groups: "1 A_tot  1" and "2 B_tot  2"
        obs = model["observables"]
        assert obs[0][1] == "A_tot"
        assert obs[0][2] == [(1.0, 1)]

    def test_parse_multi_token_mm_rate_law(self):
        model = _parse_net_file(os.path.join(DATA, "mm_tqssa.net"))
        assert model["reactions"][0][3] == "MM kcat Km"


class TestCodeGeneration:
    """Test C code generation from .net files."""

    def test_generates_valid_c(self):
        c_code = generate_rhs_c(os.path.join(DATA, "simple_decay.net"))
        assert "bngsim_codegen_rhs" in c_code
        assert "#include <math.h>" in c_code
        assert "N_SPECIES 2" in c_code
        assert "p[0]" in c_code  # parameter reference

    def test_null_product_handled(self):
        """Degradation reactions (product index 0) should not
        generate ydot[-1] references."""
        # LV.net has reaction "3 2 0 k3" (W → null)
        lv_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "benchmarks",
            "models",
            "net",
            "ode",
            "LV.net",
        )
        if not os.path.exists(lv_path):
            pytest.skip("LV.net not found")
        c_code = generate_rhs_c(lv_path)
        # Should NOT contain ydot[-1]
        assert "ydot[-1]" not in c_code

    def test_michaelis_menten_net_emits_tqssa_rate(self):
        """BNG's whitespace MM rate law must not become an unknown zero rate."""
        c_code = generate_rhs_c(os.path.join(DATA, "mm_tqssa.net"))
        assert "UNKNOWN_PARAM MM" not in c_code
        assert "sqrt(" in c_code
        assert "p[0]" in c_code  # kcat
        assert "p[1]" in c_code  # Km

    def test_model_hash_deterministic(self):
        path = os.path.join(DATA, "simple_decay.net")
        h1 = compute_model_hash(path)
        h2 = compute_model_hash(path)
        assert h1 == h2
        assert len(h1) == 16


class TestCompilation:
    """Test compilation of generated C code."""

    def test_compile_simple_decay(self):
        path = os.path.join(DATA, "simple_decay.net")
        so_path = prepare_codegen(path)
        assert so_path.exists()
        assert so_path.suffix in (".dylib", ".so", ".dll")

    def test_prepare_codegen_rejects_sbml_xml(self, tmp_path):
        xml = tmp_path / "model.xml"
        xml.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="not_a_net"/>
</sbml>
"""
        )

        with pytest.raises(ValueError, match=r"BioNetGen \.net file"):
            prepare_codegen(str(xml))


class TestSimulatorCodegenRouting:
    """Regression coverage for choosing .net vs model-based codegen."""

    def test_xml_net_path_on_non_net_model_uses_model_codegen(self, monkeypatch, tmp_path):
        import bngsim
        import bngsim._codegen as codegen_mod

        # This verifies the cc-path routing mechanics by mocking the cc codegen
        # entry points; pin to the default backend so an ambient
        # BNGSIM_CODEGEN_JIT=mir (which routes through the *_source functions)
        # does not bypass the mocks. The MIR routing is covered separately.
        monkeypatch.delenv("BNGSIM_CODEGEN_JIT", raising=False)

        class DummyModel:
            def __init__(self):
                self._core = object()
                self._net_path = ""
                self._codegen_so_path = ""

            def prepare_analytical_jacobian(self):
                # GH #145: the ODE-solve setup warms the analytical Jacobian; this
                # double exercises codegen routing only, so it is a no-op here.
                return False

        class DummyCvodeSimulator:
            def __init__(self, core):
                self.core = core

        model = DummyModel()
        xml = tmp_path / "model.xml"
        xml.write_text("<sbml/>")
        calls = []

        def fake_prepare_model_codegen(arg):
            calls.append(("model", arg))
            return tmp_path / "model_codegen.so"

        def fake_prepare_codegen(arg):
            calls.append(("net", arg))
            raise AssertionError("XML net_path should not use .net codegen")

        fake_core = types.ModuleType("bngsim._bngsim_core")
        fake_core.CvodeSimulator = DummyCvodeSimulator
        monkeypatch.setitem(sys.modules, "bngsim._bngsim_core", fake_core)
        monkeypatch.setattr(codegen_mod, "prepare_model_codegen", fake_prepare_model_codegen)
        monkeypatch.setattr(codegen_mod, "prepare_codegen", fake_prepare_codegen)

        sim = bngsim.Simulator(model, method="ode", codegen=True, net_path=str(xml))

        assert calls == [("model", model)]
        assert sim._net_path == ""
        assert sim._codegen_so_path == str(tmp_path / "model_codegen.so")
        assert model._codegen_so_path == sim._codegen_so_path


class TestCompileConfig:
    """Timeout/opt-level configuration for compile_rhs (Issue #37)."""

    def test_timeout_default_and_env(self, monkeypatch):
        from bngsim import _codegen as c

        monkeypatch.delenv("BNGSIM_CODEGEN_TIMEOUT", raising=False)
        assert c._resolve_codegen_timeout() == float(c._DEFAULT_CODEGEN_TIMEOUT)

        monkeypatch.setenv("BNGSIM_CODEGEN_TIMEOUT", "120")
        assert c._resolve_codegen_timeout() == 120.0

        # 0 (or negative) disables the timeout.
        monkeypatch.setenv("BNGSIM_CODEGEN_TIMEOUT", "0")
        assert c._resolve_codegen_timeout() is None

        # Garbage falls back to the default rather than raising.
        monkeypatch.setenv("BNGSIM_CODEGEN_TIMEOUT", "soon")
        assert c._resolve_codegen_timeout() == float(c._DEFAULT_CODEGEN_TIMEOUT)

    def test_opt_flag_size_threshold(self, monkeypatch):
        from bngsim import _codegen as c

        monkeypatch.delenv("BNGSIM_CODEGEN_OPT", raising=False)
        small = c._CODEGEN_BIG_SOURCE_BYTES - 1
        big = c._CODEGEN_BIG_SOURCE_BYTES + 1
        huge = c._CODEGEN_HUGE_SOURCE_BYTES + 1
        assert c._resolve_opt_flag("cc", small) == "-O3"
        assert c._resolve_opt_flag("cc", big) == "-O1"
        assert c._resolve_opt_flag("cl", small) == "/O2"
        assert c._resolve_opt_flag("cl", big) == "/O1"
        # GH #111 follow-up: huge sources drop to -O0 so -O1's superlinear
        # compile time can't blow the timeout and fall back to ExprTk.
        assert c._CODEGEN_HUGE_SOURCE_BYTES > c._CODEGEN_BIG_SOURCE_BYTES
        assert c._resolve_opt_flag("cc", huge) == "-O0"
        assert c._resolve_opt_flag("cl", huge) == "/Od"

    def test_opt_flag_env_override(self, monkeypatch):
        from bngsim import _codegen as c

        big = c._CODEGEN_BIG_SOURCE_BYTES + 1
        huge = c._CODEGEN_HUGE_SOURCE_BYTES + 1
        monkeypatch.setenv("BNGSIM_CODEGEN_OPT", "high")
        assert c._resolve_opt_flag("cc", big) == "-O3"
        # "high"/"none" words override the size tier in both directions.
        assert c._resolve_opt_flag("cc", huge) == "-O3"
        monkeypatch.setenv("BNGSIM_CODEGEN_OPT", "none")
        assert c._resolve_opt_flag("cc", 0) == "-O0"
        assert c._resolve_opt_flag("cl", 0) == "/Od"
        monkeypatch.setenv("BNGSIM_CODEGEN_OPT", "low")
        assert c._resolve_opt_flag("cc", 0) == "-O1"
        monkeypatch.setenv("BNGSIM_CODEGEN_OPT", "2")
        assert c._resolve_opt_flag("cc", 0) == "-O2"
        # MSVC has no /O0 or /O3.
        monkeypatch.setenv("BNGSIM_CODEGEN_OPT", "0")
        assert c._resolve_opt_flag("cl", 0) == "/Od"
        monkeypatch.setenv("BNGSIM_CODEGEN_OPT", "3")
        assert c._resolve_opt_flag("cl", 0) == "/O2"
        # Invalid override falls back to size-based default.
        monkeypatch.setenv("BNGSIM_CODEGEN_OPT", "turbo")
        assert c._resolve_opt_flag("cc", big) == "-O1"

    def test_timeout_raises_named_error(self, monkeypatch):
        """A compile timeout surfaces as a RuntimeError naming the env var,
        not a bare TimeoutExpired, and leaves no temp artifacts behind."""
        import subprocess

        from bngsim import _codegen as c

        # The compile runs through _run_compile (GH #166: process-group launch
        # so a timeout reaps the backend grandchildren). It still re-raises
        # TimeoutExpired, which compile_rhs translates to the named RuntimeError.
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

        monkeypatch.setattr(c, "_run_compile", fake_run)
        with pytest.raises(RuntimeError, match="BNGSIM_CODEGEN_TIMEOUT"):
            c.compile_rhs("int main(){return 0;}", "deadbeefcafe0001")

        leftovers = list(c.CACHE_DIR.glob("rhs_deadbeefcafe0001.*"))
        assert leftovers == [], f"temp artifacts not cleaned up: {leftovers}"

    def test_atomic_install_replaces_into_cache(self, monkeypatch, tmp_path):
        """compile_rhs builds to a process-unique temp then os.replace()s it
        into the hash-named cache path; the final .so is the cached name."""
        from bngsim import _codegen as c

        monkeypatch.setattr(c, "CACHE_DIR", tmp_path)
        path = os.path.join(DATA, "simple_decay.net")
        c_source = generate_rhs_c(path)
        so_path = c.compile_rhs(c_source, "feedfacefeedface")

        assert so_path == tmp_path / f"rhs_feedfacefeedface{so_path.suffix}"
        assert so_path.exists()
        # No temp .c or temp .so left behind.
        assert list(tmp_path.glob("rhs_feedfacefeedface.*.c")) == []


class TestCorrectness:
    """Test codegen vs ExprTk correctness through CVODE."""

    def _run_comparison(self, net_path, t_end, n_points):
        """Run ExprTk and codegen, return max abs diff."""
        from bngsim._bngsim_core import (
            CvodeSimulator,
            NetworkModel,
            SolverOptions,
            TimeSpec,
        )

        so_path = str(prepare_codegen(net_path))

        # ExprTk baseline
        m1 = NetworkModel.from_net(net_path)
        s1 = CvodeSimulator(m1)
        ts = TimeSpec()
        ts.t_start = 0.0
        ts.t_end = t_end
        ts.n_points = n_points
        r1 = s1.run(ts, SolverOptions())

        # Codegen
        m2 = NetworkModel.from_net(net_path)
        s2 = CvodeSimulator(m2)
        opts = SolverOptions()
        opts.codegen_so_path = so_path
        r2 = s2.run(ts, opts)

        sp1 = np.array(r1.species_data)
        sp2 = np.array(r2.species_data)
        return np.max(np.abs(sp1 - sp2)), r1, r2

    def test_simple_decay(self):
        path = os.path.join(DATA, "simple_decay.net")
        max_diff, r1, r2 = self._run_comparison(path, 50.0, 51)
        assert max_diff < 1e-6
        assert r1.solver_stats.n_steps == r2.solver_stats.n_steps

    def test_reversible_binding(self):
        path = os.path.join(DATA, "two_species_reversible.net")
        max_diff, r1, r2 = self._run_comparison(path, 1000.0, 101)
        assert max_diff < 1e-6

    def test_michaelis_menten_tqssa(self):
        path = os.path.join(DATA, "mm_tqssa.net")
        max_diff, r1, r2 = self._run_comparison(path, 20.0, 41)
        assert max_diff < 1e-6

    def test_analytical_solution(self):
        """Codegen matches analytical solution for simple decay."""
        path = os.path.join(DATA, "simple_decay.net")
        so_path = str(prepare_codegen(path))

        from bngsim._bngsim_core import (
            CvodeSimulator,
            NetworkModel,
            SolverOptions,
            TimeSpec,
        )

        m = NetworkModel.from_net(path)
        s = CvodeSimulator(m)
        ts = TimeSpec()
        ts.t_start = 0.0
        ts.t_end = 50.0
        ts.n_points = 51
        opts = SolverOptions()
        opts.codegen_so_path = so_path
        r = s.run(ts, opts)

        sp = np.array(r.species_data)
        t = np.array(r.time)
        A_exact = 100 * np.exp(-0.1 * t)
        max_err = np.max(np.abs(sp[:, 0] - A_exact))
        assert max_err < 1e-5, f"max error vs analytical: {max_err}"


class TestExprTkConstructs:
    """Codegen ↔ ExprTk parity for BNGL function-expression constructs that
    used to be mistranslated by ``_translate_expr``.

    Each case writes a tiny .net file with one function-rate reaction whose
    rate constant exercises one ExprTk construct, then asserts that the
    codegen and interpreted ODE paths agree at all output points.

    Most relevant case: ``(int^int)`` (e.g. ``(10^-6)``, ``(2^3)``) — these
    used to silently miscompile to bitwise XOR because both operands are
    int-typed and ``^`` is a valid int operator in C. They were the only
    silent-divergence trigger; every other broken construct caused a noisy
    compile failure that PyBNF's BngsimModel adapter swallowed via its
    try/except fallback to interpreted RHS.
    """

    NET_TEMPLATE = (
        "begin parameters\n"
        "    1 k0 0.1   # Constant\n"
        "    2 K  3.0   # Constant\n"
        "    3 n  2.0   # Constant\n"
        "end parameters\n"
        "begin species\n"
        "    1 A() 10\n"
        "end species\n"
        "begin functions\n"
        "    1 myrate() {expr}\n"
        "end functions\n"
        "begin reactions\n"
        "    1 1 0 myrate #_R1\n"
        "end reactions\n"
        "begin groups\n"
        "    1 A_obs 1\n"
        "end groups\n"
    )

    def _write_net(self, expr, tmp_path):
        net = tmp_path / "probe.net"
        net.write_text(self.NET_TEMPLATE.format(expr=expr))
        return str(net)

    def _compare(self, expr, tmp_path, t_end=5.0, n_points=11):
        from bngsim._bngsim_core import (
            CvodeSimulator,
            NetworkModel,
            SolverOptions,
            TimeSpec,
        )

        net = self._write_net(expr, tmp_path)
        so_path = str(prepare_codegen(net))

        ts = TimeSpec()
        ts.t_start = 0.0
        ts.t_end = t_end
        ts.n_points = n_points

        m1 = NetworkModel.from_net(net)
        s1 = CvodeSimulator(m1)
        r1 = s1.run(ts, SolverOptions())

        m2 = NetworkModel.from_net(net)
        s2 = CvodeSimulator(m2)
        opts = SolverOptions()
        opts.codegen_so_path = so_path
        r2 = s2.run(ts, opts)

        sp1 = np.array(r1.species_data)
        sp2 = np.array(r2.species_data)
        return float(np.max(np.abs(sp1 - sp2)))

    @pytest.mark.parametrize(
        "label,expr",
        [
            # The silent miscompile: integer-int caret (10^-6 = -16 as XOR,
            # 1e-6 as ExprTk power) is what caused flagellar_motor.net to
            # diverge by 99.82% on the Motor(state~CW) species.
            ("caret_int_neg", "k0 * A_obs * (10^-6) * 1e6"),
            ("caret_int_pos", "k0 * A_obs * (2^3) / 8.0"),
            # Caret on doubles — used to be a noisy compile failure;
            # should now translate via pow().
            ("caret_obs_int", "k0 * A_obs^2 / 10"),
            ("caret_param_int", "k0 * A_obs * (K^n) / 9.0"),
            # Nested caret inside an exponent (the nfkb.net pattern):
            # ``a^((10^c)+1)``. The outer ^ used to be replaced but the
            # inner one survived untouched.
            ("caret_nested", "k0 * A_obs * (2.0^((1^1)+1)) / 4.0"),
            # if(cond, a, b) — must become C ternary, not the C ``if`` keyword.
            ("if_branch", "k0 * if(A_obs>0, 1.0, 0.0)"),
            # abs(x) — must become fabs() so doubles aren't silently
            # truncated by the int-typed C abs().
            ("abs_double", "k0 * abs(A_obs - 5.5) / 4.5"),
            # Word-form logical operators must become && / ||.
            ("and_or", "if(A_obs>0 and (k0>0 or K>0), k0, 0)"),
            # ExprTk constants.
            ("pi_const", "k0 * (_pi/3.141592653589793)"),
            ("e_const", "k0 * (_e/2.718281828459045)"),
            # Existing translations that should keep working.
            ("ln_arg", "k0 * ln(A_obs+1) / ln(11)"),
            ("rint_arg", "k0 * rint(A_obs+0.4) / 10"),
        ],
    )
    def test_construct_parity(self, label, expr, tmp_path):
        max_diff = self._compare(expr, tmp_path)
        assert max_diff < 1e-9, (
            f"{label}: codegen ↔ interpreted disagree by {max_diff:.3e} on expr {expr!r}"
        )

    def test_null_reactant_synthesis(self, tmp_path):
        """`0 → product` reactions must not emit y[-1] in the rate term.

        Previously _rate_elementary / _rate_functional iterated all reactant
        indices including 0 (the null reactant marker), producing
        ``rate = k * y[-1]`` — out-of-bounds memory read whose value
        happened to be near 1.0 sometimes but was effectively unbounded.
        """
        from bngsim._bngsim_core import (
            CvodeSimulator,
            NetworkModel,
            SolverOptions,
            TimeSpec,
        )

        net_src = (
            "begin parameters\n"
            "    1 ksyn 1.0   # Constant\n"
            "    2 kdeg 0.1   # Constant\n"
            "end parameters\n"
            "begin species\n"
            "    1 A() 0\n"
            "end species\n"
            "begin functions\n"
            "    1 fsyn() ksyn\n"
            "end functions\n"
            "begin reactions\n"
            "    1 0 1 ksyn #_R1\n"  # elementary synthesis null → A
            "    2 0 1 fsyn #_R2\n"  # functional synthesis null → A
            "    3 1 0 kdeg #_R3\n"  # decay
            "end reactions\n"
            "begin groups\n"
            "    1 A_obs 1\n"
            "end groups\n"
        )
        net = tmp_path / "null_reactant.net"
        net.write_text(net_src)
        so_path = str(prepare_codegen(str(net)))

        # Generated C must not contain y[-1].
        c_src_path = so_path.replace(so_path[so_path.rfind(".") :], ".c")
        from pathlib import Path

        if Path(c_src_path).exists():
            assert "y[-1]" not in Path(c_src_path).read_text(), (
                "codegen emitted y[-1] for a null reactant — out-of-bounds read"
            )

        ts = TimeSpec()
        ts.t_start = 0.0
        ts.t_end = 20.0
        ts.n_points = 21

        m1 = NetworkModel.from_net(str(net))
        r1 = CvodeSimulator(m1).run(ts, SolverOptions())
        m2 = NetworkModel.from_net(str(net))
        opts = SolverOptions()
        opts.codegen_so_path = so_path
        r2 = CvodeSimulator(m2).run(ts, opts)
        sp1 = np.array(r1.species_data)
        sp2 = np.array(r2.species_data)
        assert np.max(np.abs(sp1 - sp2)) < 1e-9


class TestTfunCodegen:
    """Codegen ↔ interpreted parity for whole-function ``tfun(...)`` bodies.

    Before this work, .net codegen treated ``tfun(...)`` as an undeclared C
    function and the build silently fell back to the ExprTk interpreter.
    These tests exercise the three index kinds (implicit time, parameter,
    explicit time + step interpolation) using the canned fixtures under
    ``bngsim/tests/data``.
    """

    @staticmethod
    def _run_parity(net_path, t_end, n_points):
        from bngsim._bngsim_core import (
            CvodeSimulator,
            NetworkModel,
            SolverOptions,
            TimeSpec,
        )

        so_path = str(prepare_codegen(net_path))

        ts = TimeSpec()
        ts.t_start = 0.0
        ts.t_end = t_end
        ts.n_points = n_points

        m1 = NetworkModel.from_net(net_path)
        r1 = CvodeSimulator(m1).run(ts, SolverOptions())
        m2 = NetworkModel.from_net(net_path)
        opts = SolverOptions()
        opts.codegen_so_path = so_path
        r2 = CvodeSimulator(m2).run(ts, opts)

        sp1 = np.array(r1.species_data)
        sp2 = np.array(r2.species_data)
        return float(np.max(np.abs(sp1 - sp2))), so_path

    def test_tfun_time_indexed(self):
        """Time-indexed tfun (implicit ``time`` argument)."""
        path = os.path.join(DATA, "tfun_time_indexed.net")
        max_diff, so_path = self._run_parity(path, t_end=7.0, n_points=15)
        assert max_diff < 1e-9, f"max diff {max_diff:.3e}"
        # Generated C must call tfun_eval, not the undeclared tfun() symbol.
        from pathlib import Path

        c_path = Path(so_path).with_suffix(".c")
        if c_path.exists():
            text = c_path.read_text()
            assert "tfun_eval" in text
            assert "tfun(" not in text  # raw BNGL tfun must not survive

    def test_tfun_param_indexed(self):
        """Parameter-indexed tfun (e.g., ``tfun('dose_response.tfun', drug_conc)``)."""
        path = os.path.join(DATA, "tfun_param_indexed.net")
        max_diff, _ = self._run_parity(path, t_end=200.0, n_points=21)
        assert max_diff < 1e-9, f"max diff {max_diff:.3e}"

    def test_tfun_step(self):
        """Step interpolation via ``method=>'step'``. The interp method
        lives inside the TableFunction object; the codegen just emits the
        callback and the runtime dispatches to the right method."""
        path = os.path.join(DATA, "tfun_step_time_indexed.net")
        max_diff, _ = self._run_parity(path, t_end=3.0, n_points=31)
        assert max_diff < 1e-9, f"max diff {max_diff:.3e}"

    def test_tfun_cache_invalidates_on_data_change(self, tmp_path):
        """Editing a referenced .tfun file must change the model hash so
        the cached .so is not reused with stale interpolation data."""
        from bngsim._codegen import compute_model_hash

        # Copy fixture into tmp_path so we can edit the .tfun freely.
        net_src = (
            "begin parameters\n"
            "    1 k0 0.0\n"
            "end parameters\n"
            "begin functions\n"
            "    1 g()  tfun('g.tfun')\n"
            "end functions\n"
            "begin species\n"
            "    1 A() 1\n"
            "    2 B() 0\n"
            "end species\n"
            "begin reactions\n"
            "    1 1 1,2 g #_R1\n"
            "end reactions\n"
            "begin groups\n"
            "    1 A_tot 1\n"
            "    2 B_tot 2\n"
            "end groups\n"
        )
        net = tmp_path / "g.net"
        net.write_text(net_src)
        tfun_path = tmp_path / "g.tfun"
        tfun_path.write_text("# time g\n0 0\n1 1\n2 2\n")

        h1 = compute_model_hash(str(net))
        # Same content → same hash.
        assert h1 == compute_model_hash(str(net))

        # Edit y-values; hash must change.
        tfun_path.write_text("# time g\n0 0\n1 5\n2 9\n")
        h2 = compute_model_hash(str(net))
        assert h1 != h2

        # Reverting the data → reverting the hash.
        tfun_path.write_text("# time g\n0 0\n1 1\n2 2\n")
        assert h1 == compute_model_hash(str(net))


class TestTfunModelCodegen:
    """Same parity check as TestTfunCodegen, but for the model-based
    codegen path (``prepare_model_codegen`` / ``generate_rhs_from_model``).

    The model-based path consumes a built NetworkModel via
    ``codegen_data()`` rather than re-parsing the .net file. Function
    expressions there have already been rewritten to ``tfun_<name>()`` by
    ModelBuilder. Before this work the codegen leaked that rewritten
    identifier into C and the build failed; the fix is to dispatch
    tfun-backed functions through the runtime callback exposed via
    ``data["table_functions"]``.
    """

    @staticmethod
    def _run_parity(net_path, t_end, n_points):
        from bngsim._bngsim_core import (
            CvodeSimulator,
            NetworkModel,
            SolverOptions,
            TimeSpec,
        )
        from bngsim._codegen import prepare_model_codegen

        m_for_codegen = NetworkModel.from_net(net_path)
        so_path = prepare_model_codegen(m_for_codegen)
        assert so_path is not None, "prepare_model_codegen returned None"

        ts = TimeSpec()
        ts.t_start = 0.0
        ts.t_end = t_end
        ts.n_points = n_points

        m1 = NetworkModel.from_net(net_path)
        r1 = CvodeSimulator(m1).run(ts, SolverOptions())
        m2 = NetworkModel.from_net(net_path)
        opts = SolverOptions()
        opts.codegen_so_path = str(so_path)
        r2 = CvodeSimulator(m2).run(ts, opts)

        sp1 = np.array(r1.species_data)
        sp2 = np.array(r2.species_data)
        return float(np.max(np.abs(sp1 - sp2))), str(so_path)

    def test_tfun_time_indexed(self):
        path = os.path.join(DATA, "tfun_time_indexed.net")
        max_diff, so_path = self._run_parity(path, t_end=7.0, n_points=15)
        assert max_diff < 1e-9, f"max diff {max_diff:.3e}"
        # Make sure the rewritten tfun_<name>() identifier didn't leak
        # through into the generated C.
        from pathlib import Path

        c_path = Path(so_path).with_suffix(".c")
        if c_path.exists():
            text = c_path.read_text()
            assert "tfun_eval" in text
            assert "tfun_cumNcases" not in text

    def test_tfun_param_indexed(self):
        path = os.path.join(DATA, "tfun_param_indexed.net")
        max_diff, _ = self._run_parity(path, t_end=200.0, n_points=21)
        assert max_diff < 1e-9, f"max diff {max_diff:.3e}"

    def test_tfun_step(self):
        path = os.path.join(DATA, "tfun_step_time_indexed.net")
        max_diff, _ = self._run_parity(path, t_end=3.0, n_points=31)
        assert max_diff < 1e-9, f"max diff {max_diff:.3e}"

    def test_codegen_data_exposes_table_functions(self):
        """codegen_data() must return a table_functions list aligned with
        the runtime dispatch order."""
        from bngsim._bngsim_core import NetworkModel

        m = NetworkModel.from_net(os.path.join(DATA, "tfun_param_indexed.net"))
        d = m.codegen_data()
        tfs = d["table_functions"]
        assert len(tfs) == 1
        assert tfs[0]["name"] == "response"
        assert tfs[0]["index_kind"] == "parameter"
        # drug_conc is the second parameter (0-based idx 1) in the fixture.
        assert tfs[0]["index_param_idx"] == 1
        assert tfs[0]["index_obs_idx"] == -1


class TestRealisticModels:
    """End-to-end parity check on benchmark .net files that exercise the
    full ExprTk surface (Hill kinetics, BooleanFunction, exp/log, nested
    powers, conditional rate laws).
    """

    # Codegen consumes any .net regardless of how a model would be simulated,
    # so look in both the ode and ssa corpora before the legacy tests/data
    # fallback.
    _BENCH_ROOT = os.path.join(
        os.path.dirname(__file__), "..", "..", "benchmarks", "models", "net"
    )
    BENCH = os.path.join(_BENCH_ROOT, "ode")
    BENCH_ALT_SSA = os.path.join(_BENCH_ROOT, "ssa")

    @pytest.mark.parametrize(
        "name",
        [
            "flagellar_motor.net",  # (10^-6) used to silently XOR
            "Kholodenko_2000.net",  # MAPK_PP^n in denominator
            "oscillatory_system.net",  # if() + Kd^n + NucProt^n
            "gene_expression.net",  # if() + null-reactant synthesis
            "gene_expression_hill.net",  # NP^n / (K^n + NP^n)
            "nfkb.net",  # nested caret: a^((10^b)+1)
            "func_composition.net",  # f-of-f + if(cond&&cond, …)
        ],
    )
    def test_codegen_matches_interpreted(self, name):
        from bngsim._bngsim_core import (
            CvodeSimulator,
            NetworkModel,
            SolverOptions,
            TimeSpec,
        )

        path = os.path.join(self.BENCH, name)
        if not os.path.exists(path):
            ssa_alt = os.path.join(self.BENCH_ALT_SSA, name)
            data_alt = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", name)
            if os.path.exists(ssa_alt):
                path = ssa_alt
            elif os.path.exists(data_alt):
                path = data_alt
            else:
                pytest.skip(f"{name} not in benchmarks/models/net/{{ode,ssa}} or tests/data")

        so_path = str(prepare_codegen(path))

        ts = TimeSpec()
        ts.t_start = 0.0
        ts.t_end = 10.0
        ts.n_points = 21

        m1 = NetworkModel.from_net(path)
        r1 = CvodeSimulator(m1).run(ts, SolverOptions())
        m2 = NetworkModel.from_net(path)
        opts = SolverOptions()
        opts.codegen_so_path = so_path
        r2 = CvodeSimulator(m2).run(ts, opts)

        sp1 = np.array(r1.species_data)
        sp2 = np.array(r2.species_data)
        # Per-species relative diff; tolerate roundoff at the 1e-7 level
        # but reject the >1% gaps the silent-XOR bug used to produce.
        scale = np.maximum(np.abs(sp1), 1e-30)
        rel = float(np.max(np.abs(sp1 - sp2) / scale))
        assert rel < 1e-6, f"{name}: max rel diff {rel:.3e}"


class TestCodegenRationalLiterals:
    """Integer-literal float-ification in the C translator (codegen v9).

    ExprTk evaluates all arithmetic in ``double`` (``1/2`` == 0.5); C does
    integer division on two integer literals (``1/2`` == 0). The codegen path
    passed ExprTk strings to C verbatim, so any rate law carrying a rational
    constant silently zeroed under codegen. Surfaced by MODEL1112100000
    (1012-species WUSCHEL model): every ``Wus_*`` synthesis used a ``Sigma``
    sigmoid whose leading ``(1/2)`` codegen'd to ``0``, freezing all ``Wus``
    species at their initial value while RoadRunner and the ExprTk RHS grew
    them. The SBML loader auto-enables codegen at >=256 species, so this only
    bit large models.
    """

    def test_floatify_unit(self):
        from bngsim._codegen import _floatify_int_literals

        # rational constant → float division (the bug)
        assert _floatify_int_literals("(1/2)*x") == "(1.0/2.0)*x"
        assert _floatify_int_literals("3/4") == "3.0/4.0"
        # power exponent (pow() takes doubles)
        assert _floatify_int_literals("x^2 + 1") == "x^2.0 + 1.0"
        # identifiers and float mantissas are untouched
        assert _floatify_int_literals("A_1 + p2 + 1.5") == "A_1 + p2 + 1.5"
        # scientific notation must survive verbatim — float-ifying an exponent
        # digit would produce invalid C like ``2.5E-3.0``
        assert _floatify_int_literals("1e9 + 2.5E-3 + 6.022e23") == "1e9 + 2.5E-3 + 6.022e23"
        assert _floatify_int_literals("a*1E+5 - 7") == "a*1E+5 - 7.0"

    def test_translate_to_c_rational(self):
        from bngsim._codegen import _build_ident_lookup_model, _translate_expr_to_c

        # ((S^2+1)^(1/2)) sigmoid fragment: the (1/2) must become 1.0/2.0,
        # array subscripts introduced by substitution stay integer.
        lookup = _build_ident_lookup_model({}, {"S": "y[5]"}, {}, {})
        c = _translate_expr_to_c("(1/2)*(S^2+1)^(1/2)", lookup)
        assert "1.0/2.0" in c
        assert "y[5]" in c  # subscript untouched
        assert "(1/2)" not in c

    @staticmethod
    def _sbml_rational_functional(half_literal):
        # ∅ → P with a Functional rate ``half * kprod * (S + 1)``; the ``+1``
        # makes it a sum (not pure mass-action) so it routes Functional, and
        # ``half`` is a rational literal ``1/2``. S is a boundary constant = 4,
        # so the rate is the constant ``half*kprod*5``; with the integer-divide
        # bug ``half`` collapses to 0 and P never moves.
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="rat">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="P" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="S" compartment="c" initialConcentration="4"
               hasOnlySubstanceUnits="false" boundaryCondition="true" constant="true"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kprod" value="2" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="syn" reversible="false">
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <listOfModifiers>
          <modifierSpeciesReference species="S"/>
        </listOfModifiers>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/>
              {half_literal}
              <ci>kprod</ci>
              <apply><plus/><ci>S</ci><cn type="integer">1</cn></apply>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

    def test_codegen_rational_rate_matches_interpreted(self):
        """A Functional rate with a ``1/2`` rational gives identical P under the
        codegen and ExprTk RHS, and P actually grows (the bug froze it at 0)."""
        import bngsim
        from bngsim._codegen import prepare_model_codegen

        half = '<cn type="rational"> 1 <sep/> 2 </cn>'
        sbml = self._sbml_rational_functional(half)

        # ExprTk reference
        m_ref = bngsim.Model.from_sbml_string(sbml)
        r_ref = bngsim.Simulator(m_ref, method="ode").run(t_span=(0, 4), n_points=9)
        names = list(r_ref.species_names)
        P_ref = np.asarray(r_ref.species)[:, names.index("P")]

        # Oracle: rate = (1/2)*kprod*(S+1) = 0.5*2*5 = 5 ⇒ P(t) = 5t
        t = np.asarray(r_ref.time)
        np.testing.assert_allclose(P_ref, 5.0 * t, rtol=1e-7, atol=1e-9)
        assert P_ref[-1] > 1.0, "P should grow; rational must not zero the rate"

        # Codegen path forced via prepare_model_codegen
        m_cg = bngsim.Model.from_sbml_string(sbml)
        so = prepare_model_codegen(m_cg)
        assert so is not None
        m_cg._codegen_so_path = str(so)
        r_cg = bngsim.Simulator(m_cg, method="ode").run(t_span=(0, 4), n_points=9)
        P_cg = np.asarray(r_cg.species)[:, list(r_cg.species_names).index("P")]
        np.testing.assert_allclose(P_cg, P_ref, rtol=1e-7, atol=1e-9)

    def test_sbml_codegen_true_with_xml_net_path_uses_model_codegen(self, tmp_path):
        """Regression for GH #101: an SBML XML path passed as ``net_path`` must
        not be parsed as an empty .net model when ``codegen=True`` is requested."""
        import bngsim

        half = '<cn type="rational"> 1 <sep/> 2 </cn>'
        sbml = self._sbml_rational_functional(half)
        xml_path = tmp_path / "rat.xml"
        xml_path.write_text(sbml)

        m_ref = bngsim.Model.from_sbml(str(xml_path))
        r_ref = bngsim.Simulator(m_ref, method="ode", codegen=False).run(t_span=(0, 4), n_points=9)

        m_cg = bngsim.Model.from_sbml(str(xml_path))
        sim = bngsim.Simulator(
            m_cg,
            method="ode",
            codegen=True,
            net_path=str(xml_path),
        )
        # Codegen wired (the MIR JIT backend stashes the C source instead of a
        # .so; either proves model-based codegen, not empty-.net parsing, ran).
        assert sim._codegen_so_path or sim._codegen_c_source
        assert sim._net_path == ""

        r_cg = sim.run(t_span=(0, 4), n_points=9)
        np.testing.assert_allclose(r_cg.species, r_ref.species, rtol=1e-7, atol=1e-9)
