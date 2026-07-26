"""Shared fixtures for bngsim Python tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Test data lives in bngsim/tests/data/ — resolve via env var or relative to this file.
_DATA_DIR_ENV = os.environ.get("BNGSIM_TEST_DATA")


def pytest_configure(config: pytest.Config) -> None:
    """Stale-binary preflight (issue #125).

    The whole point of the test suite is to make true statements about the
    code. But the editable install loads a separately-built _bngsim_core that
    does not auto-rebuild (#23), so a forgotten rebuild means the suite reports
    on OLD C++ — a green run that gets committed as a correctness verdict about
    code that isn't actually running (this is what burned GH #118). Print the
    loaded binary's identity, and refuse to run against a demonstrably stale
    one. Escape hatches: BNGSIM_ALLOW_STALE_CORE=1 (proceed with a warning) or
    BNGSIM_NO_BUILD_CHECK=1 (skip the check entirely).
    """
    try:
        from bngsim import _build_provenance as bp
    except Exception:
        return  # never let the guard itself break collection
    prov = bp.gather()
    print(bp.identity_line(prov), file=sys.stderr, flush=True)
    report = bp.blocking_report(prov)
    if report:
        raise pytest.UsageError("\n" + report)


# ─── Skip audit ────────────────────────────────────────────────────────────────
#
# Sibling in spirit to the stale-binary preflight above: that guard stops the
# suite reporting on code that isn't running, this one stops it reporting on
# tests that aren't running. A skipped test and a passing test are the same
# character in the summary line, so a permanently-dead test is invisible.
#
# That is not hypothetical. test_sbml_reversible_split.py::test_copasi_abc_xml_loads
# resolved its fixture one directory level too high and had therefore NEVER run,
# on any machine, since it was written — the first draft of this audit is what
# surfaced it. Separately, three whole classes skipped on every CI leg we have
# because they imported PyBNF (lanl/bngsim#45); one had been red for months on the
# only boxes that could run it.
#
# Nothing in CI runs the full Python suite (every workflow pytest call is a
# curated file list), so the audience for this is the pre-push hook — the one
# gate that runs everything. Printing the table there means a dev sees, on every
# push, exactly what did not run.
#
# Undeclared reasons warn by default. Set BNGSIM_SKIP_AUDIT=strict to make them
# fail instead; BNGSIM_SKIP_AUDIT=off silences the block entirely. Strict is
# opt-in because the per-environment reason set is still settling — a curated CI
# leg skips a different subset than a full local run, and a guard that cries wolf
# gets disabled, which would leave us worse off than a quiet one.

# Declared skip reasons: (substring to match, why this skip is legitimate).
# A skip whose reason matches none of these is reported as undeclared. Adding an
# entry is the point — it forces a new permanent skip to be justified in a diff
# rather than blending into the summary count.
_DECLARED_SKIPS: tuple[tuple[str, str], ...] = (
    # Build-configuration variants — the feature is genuinely absent from this build.
    ("without the MIR backend", "MIR JIT is off unless -DBNGSIM_ENABLE_MIR=ON"),
    ("KLU not compiled", "KLU-off builds are a supported configuration"),
    ("requires a build without SuiteSparse/KLU", "inverse of the above; KLU-off builds only"),
    ("LAPACK-dense not built", "LAPACK is optional; CMake degrades to the reference solver"),
    ("RuleMonkey compiled in", "inverse-condition test; runs only on RuleMonkey-off builds"),
    ("RuleMonkey not compiled in", "RuleMonkey is a build-time opt-in"),
    # Optional / developer-only Python dependencies.
    ("could not import", "optional extra (h5py, jax, pandas, sympy, xarray, ...) absent"),
    ("roadrunner", "DEVELOPER-ONLY reference engine; never a base dependency"),
    ("scipy", "optional extra"),
    ("antimony", "optional extra; loaders fall back to SBML"),
    # External tools and corpora that are not vendored.
    ("BNG2.pl", "external perl toolchain, not a bngsim dependency"),
    ("biomodels", "BioModels corpus is fetched, not vendored ($BIOMODELS_SBML_DIR)"),
    ("benchmark", "benchmark corpus lives outside the packaged tree"),
    ("abc.xml not at", "fixture in a sibling PyBNF checkout; dev-only"),
    # Source-tree vs installed-wheel context.
    ("installed wheel", "source-tree-only guard, correctly inert against a wheel"),
    ("source root", "version-consistency check needs the source tree"),
    ("CMake", "CMakeCache cross-checks need a configured build dir"),
    ("explicitly bypassed via env", "the escape hatch reporting that it was used"),
    # Missing .net / .xml fixtures. Deliberately last and deliberately narrow:
    # this is the category that rots silently, so it matches the exact phrasings
    # in use rather than a blanket "not found".
    ("not present", "optional fixture absent from this checkout"),
    ("not found", "optional fixture absent from this checkout"),
    ("not available", "optional fixture absent from this checkout"),
)


def _skip_reason(report: object) -> str:
    """Pull the human reason out of a skip report, minus pytest's prefix."""
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        reason = str(longrepr[2])
    else:
        reason = str(longrepr or "")
    return reason[len("Skipped: ") :] if reason.startswith("Skipped: ") else reason


def _audit_skips(terminalreporter: object) -> tuple[dict[str, int], dict[str, int]]:
    """Return (reason -> count, undeclared subset thereof).

    Recomputed by each hook rather than shared through the stash: conftest's
    pytest_sessionfinish can run *before* the terminal reporter's (which is what
    invokes pytest_terminal_summary), so anything the summary stashes is not yet
    there when the exit code is decided.
    """
    counts: dict[str, int] = {}
    for report in getattr(terminalreporter, "stats", {}).get("skipped", []):
        reason = _skip_reason(report)
        counts[reason] = counts.get(reason, 0) + 1
    undeclared = {
        reason: n
        for reason, n in counts.items()
        if not any(pattern.lower() in reason.lower() for pattern, _ in _DECLARED_SKIPS)
    }
    return counts, undeclared


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(
    terminalreporter: object, exitstatus: object, config: pytest.Config
) -> None:
    """Print what did not run, and flag reasons nobody has declared."""
    mode = os.environ.get("BNGSIM_SKIP_AUDIT", "warn").lower()
    if mode == "off":
        return
    counts, undeclared = _audit_skips(terminalreporter)
    if not counts:
        return

    write = terminalreporter.write_line  # type: ignore[attr-defined]
    terminalreporter.write_sep("─", "skip audit")  # type: ignore[attr-defined]
    for reason, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        mark = "??" if reason in undeclared else "  "
        write(f" {mark} {n:>3}  {reason[:96]}")
    if undeclared:
        write("")
        write(f" {len(undeclared)} undeclared skip reason(s), marked ?? above.")
        write(" Add each to _DECLARED_SKIPS in python/tests/conftest.py with a rationale,")
        write(" or fix the test so it runs. A skip nobody declared is usually a test that")
        write(" stopped running without anyone deciding it should.")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the run for undeclared skips under BNGSIM_SKIP_AUDIT=strict."""
    if os.environ.get("BNGSIM_SKIP_AUDIT", "warn").lower() != "strict":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    _, undeclared = _audit_skips(reporter)
    # Only escalate a run that would otherwise pass; a real failure keeps its
    # own exit code, which is the more actionable one.
    if undeclared and exitstatus == 0:
        session.exitstatus = 1


@pytest.fixture
def data_dir() -> Path:
    """Path to the C++ test data directory (shared with Phase A tests).

    Set BNGSIM_TEST_DATA env var to override (useful when running tests
    from outside the source tree).
    """
    if _DATA_DIR_ENV:
        d = Path(_DATA_DIR_ENV)
    else:
        # Walk up from python/tests/ → python/ → bngsim/ then into tests/data/
        d = Path(__file__).resolve().parent.parent.parent / "tests" / "data"
    assert d.is_dir(), f"Test data directory not found: {d}"
    return d


@pytest.fixture
def simple_decay_net(data_dir: Path) -> Path:
    """Path to simple_decay.net."""
    return data_dir / "simple_decay.net"


@pytest.fixture
def reversible_net(data_dir: Path) -> Path:
    """Path to two_species_reversible.net."""
    return data_dir / "two_species_reversible.net"


@pytest.fixture
def time_func_net(data_dir: Path) -> Path:
    """Path to time_dependent_func.net."""
    return data_dir / "time_dependent_func.net"


@pytest.fixture
def fixed_species_net(data_dir: Path) -> Path:
    """Path to fixed_species.net."""
    return data_dir / "fixed_species.net"


@pytest.fixture
def const_expr_net(data_dir: Path) -> Path:
    """Path to const_expr_setparam.net."""
    return data_dir / "const_expr_setparam.net"


@pytest.fixture
def expr_param_net(data_dir: Path) -> Path:
    """Path to expr_param_species.net."""
    return data_dir / "expr_param_species.net"


@pytest.fixture
def exprtk_reserved_words_net(data_dir: Path) -> Path:
    """Path to exprtk_reserved_words.net (issue #18 regression)."""
    return data_dir / "exprtk_reserved_words.net"


@pytest.fixture
def t_as_observable_net(data_dir: Path) -> Path:
    """Path to t_as_observable.net (issue #24 regression)."""
    return data_dir / "t_as_observable.net"


@pytest.fixture
def reactions_text_block_net(data_dir: Path) -> Path:
    """Path to reactions_text_block.net (issue #13 regression)."""
    return data_dir / "reactions_text_block.net"


@pytest.fixture
def obs_zero_arg_call_net(data_dir: Path) -> Path:
    """Path to obs_zero_arg_call.net (issue #28 regression)."""
    return data_dir / "obs_zero_arg_call.net"


@pytest.fixture
def obs_zero_arg_call_sens_net(data_dir: Path) -> Path:
    """Path to obs_zero_arg_call_sens.net (issue #28 on the codegen path)."""
    return data_dir / "obs_zero_arg_call_sens.net"


@pytest.fixture
def sign_as_parameter_net(data_dir: Path) -> Path:
    """Path to sign_as_parameter.net (sign-collision regression)."""
    return data_dir / "sign_as_parameter.net"


@pytest.fixture
def mratio_overflow_net(data_dir: Path) -> Path:
    """Path to test_mratio_overflow.net (issue #42 regression).

    Generated by BNG2.pl from BNGL-Models/my_models/ode/test_Mratio_1.bngl.
    The parameter block computes a confluent-hypergeometric mean/sdev with
    `a=-1000, b=9001, z=-10000` — the regime where bngsim's previous naive
    power-series mratio overflowed to nan.
    """
    return data_dir / "test_mratio_overflow.net"


@pytest.fixture
def homodimer_ssa_net(data_dir: Path) -> Path:
    """Path to homodimer_ssa.net."""
    return data_dir / "homodimer_ssa.net"


@pytest.fixture
def fractional_ssa_net(data_dir: Path) -> Path:
    """Path to fractional_ssa.net."""
    return data_dir / "fractional_ssa.net"


@pytest.fixture
def saturation_net(data_dir: Path) -> Path:
    """Path to saturation.net."""
    return data_dir / "saturation.net"


@pytest.fixture
def ssa_abc_net(data_dir: Path) -> Path:
    """Path to ssa_abc.net."""
    return data_dir / "ssa_abc.net"


@pytest.fixture
def mm_tqssa_net(data_dir: Path) -> Path:
    """Path to mm_tqssa.net (Michaelis-Menten tQSSA test)."""
    return data_dir / "mm_tqssa.net"


@pytest.fixture
def sat_rewrite_net(data_dir: Path) -> Path:
    """Path to sat_rewrite.net (legacy Sat → Functional loader rewrite fixture)."""
    return data_dir / "sat_rewrite.net"


@pytest.fixture
def hill_rewrite_net(data_dir: Path) -> Path:
    """Path to hill_rewrite.net (legacy Hill → Functional loader rewrite fixture)."""
    return data_dir / "hill_rewrite.net"


@pytest.fixture
def tfun_time_indexed_net(data_dir: Path) -> Path:
    """Path to tfun_time_indexed.net (tfun indexed by time)."""
    return data_dir / "tfun_time_indexed.net"


@pytest.fixture
def tfun_param_indexed_net(data_dir: Path) -> Path:
    """Path to tfun_param_indexed.net (tfun indexed by parameter)."""
    return data_dir / "tfun_param_indexed.net"


@pytest.fixture
def tfun_step_time_indexed_net(data_dir: Path) -> Path:
    """Path to tfun_step_time_indexed.net (step-interpolated tfun)."""
    return data_dir / "tfun_step_time_indexed.net"


@pytest.fixture
def tfun_uppercase_time_net(data_dir: Path) -> Path:
    """tfun_uppercase_time.net — `.tfun` header has `# Time  cumNcases()`
    and the tfun call passes `Time` as index. Regression for GH #35:
    bngsim should honor BNG's case-insensitive `time`/`t` and trailing
    `()` stripping conventions in `.tfun` headers.
    """
    return data_dir / "tfun_uppercase_time.net"


@pytest.fixture
def tfun_paren_param_net(data_dir: Path) -> Path:
    """tfun_paren_param.net — `.tfun` column-1 header is `drug_conc()`
    (with empty parens) but the tfun call passes the bare `drug_conc`
    parameter. Regression for GH #35: trailing `()` stripping must apply
    to param-indexed tfun headers too, not just time.
    """
    return data_dir / "tfun_paren_param.net"


@pytest.fixture
def wrap_single_net(data_dir: Path) -> Path:
    """Path to wrap_single.net (wrapper-form tfun: `(tfun('drive') + 5)/k_scale`).

    Regression fixture for GH #33. The .net came from BioNetGen 2.9.3 on
    wrap_single.bngl; sister `drive.tfun` lives in the same directory so the
    loader's relative-path resolver finds it.
    """
    return data_dir / "wrap_single.net"


@pytest.fixture
def cumncases_tfun(data_dir: Path) -> Path:
    """Path to cumNcases.tfun data file."""
    return data_dir / "cumNcases.tfun"


@pytest.fixture
def dose_response_tfun(data_dir: Path) -> Path:
    """Path to dose_response.tfun data file."""
    return data_dir / "dose_response.tfun"


@pytest.fixture
def nfsim_xml(data_dir: Path) -> Path:
    """Path to NFsim simple_system.xml test file."""
    return data_dir / "nfsim" / "simple_system.xml"


@pytest.fixture
def nfsim_funccols_xml(data_dir: Path) -> Path:
    """XML with a global function and a composite (function-of-function).

    `phos_ratio() = Xp/Xtot` references observables only (a GlobalFunction);
    `phos_percent() = 100*phos_ratio()` references another function (a
    CompositeFunction). Used to verify that both kinds surface as Result
    expression columns.
    """
    return data_dir / "nfsim" / "funccols.xml"


@pytest.fixture
def nfsim_param_prop_xml(data_dir: Path) -> Path:
    """XML mirroring simple_system but with derived <Parameter expr=...> chains.

    Used to verify set_param() propagation through dependent parameters
    (issue #20). The new params introduce: kon = kon_base*kon_scale,
    use_fast = if(kon_scale>=threshold,1,0), kon_eff = kon*(1-use_fast)+kon*100*use_fast.
    """
    return data_dir / "nfsim" / "param_propagation.xml"


@pytest.fixture
def nfsim_seed_concentration_xml(data_dir: Path) -> Path:
    """XML where seed-species concentrations reference parameters.

    Used to verify pre-init set_param() rewrites `<Species concentration="X">`
    through the override-resolved namespace before NFsim creates agents
    (issue #29). X_init=5000 drives X(p~0,y); Y_init=500 drives Y(x);
    X_total=X_init+Y_init exercises propagation through a derived parameter.
    """
    return data_dir / "nfsim" / "seed_concentration_param.xml"


@pytest.fixture
def nfsim_switchable_rate_xml(data_dir: Path) -> Path:
    """A(b)+B(b)<->A.B with a switchable rate and a derived seed amount (GH #44).

    ``kf`` (the binding rate) is 0 at load, so the bind rule would be dropped by
    NFsim's parser unless the session keeps zero-rate rules — this exercises Bug
    1 (a post-init ``set_param('kf', ...)`` must be able to activate the rule).
    Both seed species use ``concentration="Ntot"`` where ``Ntot = 100*scale``, a
    *derived* parameter — this exercises Bug 2 (a pre-init ``set_param('scale',
    ...)`` must re-derive the seed-species amount, matching NFsim). Shared by the
    NFsim and RuleMonkey session tests.
    """
    return data_dir / "nfsim" / "switchable_rate.xml"


@pytest.fixture
def fractional_init_xml(data_dir: Path) -> Path:
    """Path to XML fixture with non-integer initial species count."""
    return data_dir / "nfsim" / "fractional_init.xml"


@pytest.fixture
def nfsim_malformed_xml(data_dir: Path) -> Path:
    """Path to malformed NFsim XML (triggers parse error)."""
    return data_dir / "nfsim" / "malformed.xml"


@pytest.fixture
def nfsim_empty_model_xml(data_dir: Path) -> Path:
    """XML from a BNGL file with only a parameters block — no molecule
    types, observables, or reactions, but with a ``simulate({method=>"nf",...})``
    action. BNG2.pl emits a ``.gdat`` with only the ``# time`` header and
    the requested time rows.

    Generated from ``BNGL-Models/my_models/nf/BSA_v1.bngl`` (an unfinished
    BNGL file that snuck into a corpus sweep). Used to verify that
    bngsim's NFsim path matches BNG2.pl's vacuous-success behavior and
    that ``Result.has_simulation_data`` reports ``False`` for the
    resulting empty Result — see issue #40 B-2.
    """
    return data_dir / "nfsim" / "empty_model.xml"


@pytest.fixture
def nfsim_composite_local_deps_xml(data_dir: Path) -> Path:
    """XML whose rate law is a CompositeFunction with local-function deps.

    Minimal BLBR-style model: A flips s 0->1 at a rate that depends on a
    molecule-scope observable Atot(a). BNG2.pl emits `a` as type="Local"
    on the synthesized `_rateLaw1` composite, so a scope-free
    evaluateOn(nullptr,nullptr,...) call hits the local-deps branch.

    Used by the post-Session-29 audit-closure tests: the bngsim probe in
    `resolve_output_functions` (src/nfsim_simulator.cpp) catches the
    resulting std::runtime_error and silently skips the composite. Before
    the Session-29 carry the same path called exit(1) and killed the
    process; see bngsim/dev/notes/session29_regression_audit.md.
    """
    return data_dir / "nfsim" / "composite_with_local_deps.xml"


@pytest.fixture
def nfsim_composite_reactant_count_dep_xml(data_dir: Path) -> Path:
    """XML whose rate law is a CompositeFunction with a reactant-count dep.

    Minimal model: `A(b) + A(b) -> A(b!1).A(b!1)` at rate `reactant_1()*k`.
    `reactant_1()` is declared as an empty local function (BNG emits it as a
    Function with an empty Expression, which the loader skips) and is the
    built-in reactant-count reference. The synthesized `_rateLaw1` composite
    therefore has n_lfs==0 but n_reactantCounts==1.

    GH #116: a scope-free evaluateOn(nullptr,nullptr,nullptr,0) — the probe in
    `resolve_output_functions` (src/nfsim_simulator.cpp) — used to pass the
    local-function guard (n_lfs==0) and then NULL-deref the reactant-count
    array, a native SIGSEGV during NfsimSession.initialize(). evaluateOn now
    throws a catchable error for a missing reactant-count context, so the probe
    skips the composite. Minimal isolation of actin_branch_forFitToData.
    """
    return data_dir / "nfsim" / "composite_with_reactant_count_dep.xml"


@pytest.fixture
def nfsim_tfun_xml(data_dir: Path) -> Path:
    """Path to NFsim TFUN XML fixture."""
    return data_dir / "nfsim" / "tfun_test" / "tfun_simple.xml"


@pytest.fixture
def nfsim_tfun_new_format_dir(data_dir: Path) -> Path:
    """Path to the new-format NFsim TFUN fixture directory."""
    return data_dir / "nfsim" / "tfun_new_format"


@pytest.fixture
def nfsim_compartment_xml(data_dir: Path) -> Path:
    """Path to NFsim XML fixture containing compartments (cBNGL smoke)."""
    return data_dir / "nfsim" / "compartment_test" / "simple_compartment_system.xml"


@pytest.fixture
def nfsim_sym_state_xml(data_dir: Path) -> Path:
    """XML with two same-named state-bearing components on one MoleculeType.

    `L(r~u~c~g, r~u~c~g)` — exercises NFsim's symmetric-site renaming
    (internally `r1`/`r2`) on a stateful equivalency class. Used to verify
    the species API resolves bare class-original names like `L(r~u,r~u)`
    (issue #21).
    """
    return data_dir / "nfsim" / "sym_state_sites.xml"


@pytest.fixture
def nfsim_sym_stateless_xml(data_dir: Path) -> Path:
    """XML with two same-named stateless components on one MoleculeType.

    `L(r,r)` — exercises the species API on a stateless equivalency class.
    """
    return data_dir / "nfsim" / "sym_stateless_sites.xml"
