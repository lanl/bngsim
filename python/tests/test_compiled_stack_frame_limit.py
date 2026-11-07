"""Issue #1566: a codegen library may not hold a function whose stack frame can
overflow the stack of the thread that calls it.

At -O0 a compiler gives every temporary in a function its own stack slot, so a
flat function's frame grows with the function. ``compile_rhs`` compiles an
unchunked source past ``_CODEGEN_HUGE_SOURCE_BYTES`` at -O0, and
BIOMD0000000471's sensitivity build is one: the library built in about 100 s,
its ``bngsim_dfdp`` reserved 12,751,024 bytes, and the first ``run()`` segfaulted
on the macOS main thread, whose stack is 8 MiB. Every gcc/clang compile now asks
the compiler to report a function past ``_CODEGEN_MAX_FRAME_BYTES``, and a
library with one is refused instead of installed.

The tests shrink the limit and the size tiers so a source of a few thousand lines
stands in for a corpus model's tens of MB. The same flat function is refused at
-O0 and installed at -O1, which reuses stack slots, so the guard is shown to
refuse the unsafe build and not every build. Both compilers' report formats are
pinned as text too, since each CI leg runs only one of them.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from bngsim import Model, Simulator
from bngsim import _codegen as cg

DATA = os.environ.get("BNGSIM_TEST_DATA") or os.path.join(
    os.path.dirname(__file__), "..", "..", "tests", "data"
)


def _compiler() -> list[str] | None:
    try:
        return cg._find_c_compiler()
    except Exception:  # noqa: BLE001 - no compiler is a skip, not an error
        return None


_CC = _compiler()
requires_gcc_clang = pytest.mark.skipif(
    _CC is None or _CC[0].lower().endswith("cl"),
    reason=(
        "no C compiler"
        if _CC is None
        else "gcc/clang only: MSVC reports no stack frame size outside /analyze"
    ),
)

#: The stand-in limit: an -O0 frame of the flat function below is one 8-byte slot
#: per temporary, 16 KB for 2,000; at -O1 it is under 1 KB on gcc and clang.
_LIMIT = 8 * 1024
_N_TEMPORARIES = 2_000


def _flat_function(name: str = "flat", n: int = _N_TEMPORARIES) -> str:
    """One straight-line function with ``n`` named temporaries, the shape of a
    generated Jacobian or ``bngsim_dfdp`` case."""
    lines = [f"void {name}(const double* y, const double* p, double* out) {{"]
    for i in range(n):
        lines.append(f"    double t{i} = y[{i % 50}] * p[{i % 7}] + y[{(i * 3) % 50}];")
        lines.append(f"    out[{i % 100}] += t{i};")
    lines.append("}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def small_limit(monkeypatch, tmp_path):
    """The stand-in limit and a private cache; the size tiers stay as they are
    until a test moves them."""
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cg, "_CODEGEN_MAX_FRAME_BYTES", _LIMIT)
    monkeypatch.delenv("BNGSIM_CODEGEN_OPT", raising=False)
    monkeypatch.delenv("BNGSIM_CODEGEN_TIMEOUT", raising=False)
    return tmp_path


def _tiers_put_source_at(monkeypatch, source: str, flag: str) -> None:
    """Move the size tiers so ``source`` compiles at ``flag`` by size, as a
    corpus model's tens of MB do at the real thresholds."""
    n = len(source)
    if flag == "-O0":
        monkeypatch.setattr(cg, "_CODEGEN_BIG_SOURCE_BYTES", n // 4)
        monkeypatch.setattr(cg, "_CODEGEN_HUGE_SOURCE_BYTES", n // 2)
    elif flag == "-O1":
        monkeypatch.setattr(cg, "_CODEGEN_BIG_SOURCE_BYTES", n // 2)
        monkeypatch.setattr(cg, "_CODEGEN_HUGE_SOURCE_BYTES", n * 2)
    else:
        raise ValueError(flag)
    assert cg._resolve_opt_flag("cc", n) == flag


class TestTheCompilerIsAskedForTheReport:
    def test_every_gcc_clang_command_carries_the_limit(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cg, "_CODEGEN_MAX_FRAME_BYTES", 12_345)
        cmd = cg._build_compile_cmd(tmp_path / "a.c", tmp_path / "a.so", "-O2", ["cc"])
        assert "-Wframe-larger-than=12345" in cmd

    def test_msvc_is_not_given_a_flag_it_does_not_take(self, tmp_path):
        cmd = cg._build_compile_cmd(tmp_path / "a.c", tmp_path / "a.dll", "/O2", ["cl"])
        assert not any("frame-larger-than" in tok for tok in cmd)

    def test_the_shard_compile_carries_it_too(self, monkeypatch, tmp_path):
        seen: list[list[str]] = []

        class _Done:
            returncode = 0
            stderr = ""

        def record(cmd, **kwargs):
            seen.append(cmd)
            (tmp_path / "u.o").write_bytes(b"")
            return _Done()

        monkeypatch.setattr(cg, "_run_compile", record)
        monkeypatch.setattr(cg, "_CODEGEN_MAX_FRAME_BYTES", 54_321)
        cg._compile_one_object(["cc"], "-O2", tmp_path, "u.c", "u.o", None)
        assert "-Wframe-larger-than=54321" in seen[0]


class TestTheReportIsRead:
    """The refusal turns on the option tag both compilers print; the function and
    the size are read for the message. Text from each, as each prints it."""

    def test_clang(self):
        err = (
            "big.c:1:8: warning: stack frame size (12751024) exceeds limit (4194304) "
            "in 'bngsim_dfdp' [-Wframe-larger-than]\n"
            "    1 | static void bngsim_dfdp(int iP, double t, const double* y,\n"
            "big.c:9:8: warning: stack frame size (12751008) exceeds limit (4194304) "
            "in 'bngsim_dfdp_term_scale' [-Wframe-larger-than]\n"
            "2 warnings generated.\n"
        )
        assert cg._oversized_frames(err) == [
            ("bngsim_dfdp", 12_751_024),
            ("bngsim_dfdp_term_scale", 12_751_008),
        ]

    def test_clang_before_13(self):
        err = (
            "a.c:3:6: warning: stack frame size of 9321232 bytes in function "
            "'bngsim_codegen_jac_sparse' [-Wframe-larger-than=]\n"
        )
        assert cg._oversized_frames(err) == [("bngsim_codegen_jac_sparse", 9_321_232)]

    @pytest.mark.parametrize("quote", [("'", "'"), ("‘", "’")])
    def test_gcc_names_the_function_on_the_line_above(self, quote):
        lq, rq = quote
        err = (
            f"a.c: In function {lq}flat{rq}:\n"
            "a.c:4002:1: warning: the frame size of 16032 bytes is larger than "
            "8192 bytes [-Wframe-larger-than=]\n"
        )
        assert cg._oversized_frames(err) == [("flat", 16_032)]

    def test_a_report_in_words_it_cannot_read_still_counts(self):
        err = "a.c:1:1: avertissement: cadre de pile trop grand [-Wframe-larger-than=]\n"
        assert cg._oversized_frames(err) == [("?", None)]

    def test_other_warnings_refuse_nothing(self):
        err = "a.c:1:1: warning: unused variable 'x' [-Wunused-variable]\n"
        assert cg._oversized_frames(err) == []
        cg._refuse_oversized_frames(err, "-O3", "a source")


@requires_gcc_clang
class TestCompileRhs:
    def test_a_flat_function_past_the_limit_at_O0_is_not_installed(self, small_limit, monkeypatch):
        """The reported path: the size tier picks -O0, the frame grows with the
        function, and the library used to be installed for a run to crash on."""
        source = _flat_function()
        _tiers_put_source_at(monkeypatch, source, "-O0")

        with pytest.raises(RuntimeError) as err:
            cg.compile_rhs(source, "f1566a00")

        message = str(err.value)
        assert "issue #1566" in message
        assert "flat (" in message, message
        assert "at -O0" in message and "BNGSIM_CODEGEN_OPT=1" in message
        assert not isinstance(err.value, cg._FrameTooLarge), (
            "compile_rhs raises a plain RuntimeError, as for a failed compile"
        )
        assert cg.get_cached_so("f1566a00") is None
        assert sorted(p.name for p in small_limit.iterdir()) == [], "nothing left behind"

    def test_the_same_function_at_O1_is_installed(self, small_limit, monkeypatch):
        """-O1 reuses stack slots, so the same function fits: the guard refuses
        the unsafe build, not the model."""
        source = _flat_function()
        _tiers_put_source_at(monkeypatch, source, "-O1")

        so_path = cg.compile_rhs(source, "f1566a01")

        assert so_path.exists()
        assert cg.get_cached_so("f1566a01") == so_path

    def test_a_shard_past_the_limit_refuses_the_whole_library(self, small_limit, monkeypatch):
        """A chunked source compiles in parallel shards (GH #160); a shard over
        the limit refuses the library as the serial compile does."""
        monkeypatch.setenv("BNGSIM_CODEGEN_OPT", "0")
        monkeypatch.setenv("BNGSIM_CODEGEN_JOBS", "2")
        monkeypatch.setattr(cg, "_SHARD_UNIT_BLOCKS", 1)
        monkeypatch.setattr(cg, "_available_memory_bytes", lambda: None)
        blocks = []
        for name in ("blk_small", "blk_flat"):
            body = _flat_function(name, 4 if name == "blk_small" else _N_TEMPORARIES)
            blocks += [cg._SHARD_BLOCK_OPEN, body.rstrip("\n"), cg._SHARD_BLOCK_CLOSE]
        source = "\n".join(
            [
                cg._CHUNK_MARKER,
                "void blk_small(const double* y, const double* p, double* out);",
                "void blk_flat(const double* y, const double* p, double* out);",
                *blocks,
                "void drive(const double* y, const double* p, double* out) {",
                "    blk_small(y, p, out);",
                "    blk_flat(y, p, out);",
                "}",
                "",
            ]
        )
        assert cg._split_sharded_source(source) is not None

        with pytest.raises(RuntimeError, match=r"shard .* the stack frame of blk_flat \("):
            cg.compile_rhs(source, "f1566a02")

        assert cg.get_cached_so("f1566a02") is None


@requires_gcc_clang
class TestARunIsRefusedNotCrashed:
    """Through Simulator: a refused library is a failed build, so a plain run
    falls back to the interpreted RHS as on any failed build, ``codegen=True``
    says why it could not be honoured, and a sensitivity run, which needs the
    library, refuses with the reason. The stand-in limit is below what any -O0
    function of this model reserves."""

    @pytest.fixture
    def model_at_O0(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(cg, "_CODEGEN_MAX_FRAME_BYTES", 32)
        monkeypatch.setenv("BNGSIM_CODEGEN_OPT", "0")
        monkeypatch.delenv("BNGSIM_CODEGEN_JIT", raising=False)
        monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)
        monkeypatch.delenv("BNGSIM_CODEGEN_STRICT", raising=False)
        return lambda: Model.from_net(os.path.join(DATA, "simple_decay.net"))

    def test_codegen_true_names_the_reason(self, model_at_O0):
        with pytest.raises(RuntimeError, match=r"issue #1566"):
            Simulator(model_at_O0(), method="ode", codegen=True)

    def test_a_sensitivity_run_names_the_reason(self, model_at_O0):
        with pytest.raises(RuntimeError, match=r"issue #1566") as err:
            Simulator(model_at_O0(), method="ode", sensitivity_params=["k1"])
        assert "differentiated" not in str(err.value), (
            "a refused library is a failed build, not a model that cannot be differentiated"
        )

    def test_a_plain_run_falls_back_and_runs(self, model_at_O0, monkeypatch):
        monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
        model = model_at_O0()
        sim = Simulator(model, method="ode")
        assert not getattr(model, "_codegen_so_path", ""), "no library was attached"
        result = sim.run(t_span=(0.0, 10.0), n_points=3)
        reference = Simulator(model_at_O0(), method="ode", codegen=False).run(
            t_span=(0.0, 10.0), n_points=3
        )
        np.testing.assert_allclose(result.species, reference.species, rtol=1e-6)
