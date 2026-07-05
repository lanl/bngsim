"""Shared adapters for the bng_parity timing+parity tracks (BNGsim vs the legacy
BNG2.pl/run_network/NFsim stack) — ODE, SSA, and network-free (NF).

The bng analogue of ``rr_parity/_rr_common.py``. The reference engine here is NOT
a Python library but the legacy ``BNG2.pl`` → ``run_network`` / ``NFsim`` CLI
stack, so the adapter shells out. Each track is structured around ONE shared
BNG2.pl artifact both engines then consume:

  * **ODE / SSA** — the reaction-network ``.net`` (``generate_network``). BNGsim
    loads it in-process; the legacy ``run_network`` binary runs ``-p cvode`` (ODE)
    or ``-p ssa`` (SSA) on the same ``.net``.
  * **NF** — the BNG-XML (``writeXML``; a network-free model has no ``.net``).
    BNGsim drives it through ``NfsimSession``; the legacy ``NFsim`` binary reads
    the same ``.xml``.

The deterministic (ODE) comparison is over the network species (``.cdat``, by
index); the stochastic (SSA/NF) comparison is over the observables (``.gdat``, by
NAME) of an N-replicate ensemble — NF has no species, and observables-by-name is
the uniform, robust axis for both. The ODE section below documents the original
ODE track; the SSA/NF adapters (``bn_ssa_net``/``bn_nf_xml`` +
``run_network_ssa``/``nfsim_run``) mirror it to the stochastic cost model (a
per-model load + a per-replicate ensemble, no warm-reuse of one solve).

For the ODE track the comparison is structured around ONE shared artifact: the
reaction network ``.net`` BNG2.pl generates from a BNGL model. Both engines then
integrate that **byte-identical** network —

  * **BNGsim** loads the ``.net`` in-process (``Model.from_net`` →
    ``Simulator(method="ode").run``), so its build phases (network load, analytical
    Jacobian derivation, RHS codegen) and a cold→warm integration split are timed
    at their own boundaries, exactly like the rr_parity SBML adapter.
  * **run_network** (the legacy CVODE binary, ``$BNGPATH/bin/run_network``)
    integrates the SAME ``.net`` as a direct subprocess. Each invocation is a fresh
    process — there is no warm-solver reuse — so we measure its per-call wall plus
    its OWN self-reported ``Initialization``/``Propagation`` CPU split (printed on
    stdout), the legacy analogue of an integrator's internal timing.

Network generation (BNG2.pl) is the shared per-model build *prefix*: the same
``.net`` feeds both engines, so it is measured once and attributed to neither
integrator's headline. This keeps the integration comparison apples-to-apples on
the network BNG2.pl produced.

Engine-agnostic helpers (the kill-on-overrun ``schedule``, ``hardware_info``,
``sundials_version``, ``_integrate_stats``, ``_warm_rep_count``,
``LINEAR_SOLVER_NAMES``) are reused verbatim from ``rr_parity._rr_common`` so all
three suites share ONE scheduler and ONE timing-stats convention.
"""

from __future__ import annotations

import contextlib
import keyword
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Reuse the engine-agnostic surface from rr_parity._rr_common. That module has
# NO top-level ``import roadrunner`` (RoadRunner is imported lazily inside its
# rr_* adapters only), so importing it here never pulls RoadRunner into a bng
# run. rr_parity is an implicit namespace package on parity_checks/'s path.
# --------------------------------------------------------------------------- #
_PARITY_CHECKS = Path(__file__).resolve().parent.parent
if str(_PARITY_CHECKS) not in sys.path:
    sys.path.insert(0, str(_PARITY_CHECKS))

from rr_parity import _rr_common as _rc  # noqa: E402

schedule = _rc.schedule
hardware_info = _rc.hardware_info
sundials_version = _rc.sundials_version
_integrate_stats = _rc._integrate_stats
_ensemble_stats = _rc._ensemble_stats
_warm_rep_count = _rc._warm_rep_count
LINEAR_SOLVER_NAMES = _rc.LINEAR_SOLVER_NAMES

# Shared CVODE tolerance forced on BOTH engines. Unlike rr_parity's tight
# SBML-appropriate 1e-9/1e-12, the BNGL corpus runs in particle (count) space
# where BNG2.pl's own historical default is 1e-8; we adopt that so bngsim and
# run_network integrate at the tolerance the legacy stack actually uses. Per-job
# ``tol`` overrides (ill-conditioned IVPs) tighten this identically on both
# engines, mirroring the parity_sweep TOL_OVERRIDES.
DEFAULT_ATOL = 1e-8
DEFAULT_RTOL = 1e-8

# Output points when an ode action omits n_steps/n_output_steps (run_network
# requires a step count; BNG defaults a bare simulate to 1, which is a degenerate
# 2-point trajectory — useless for a parity check, so we sample a real curve).
DEFAULT_N_STEPS = 50


# --------------------------------------------------------------------------- #
# Horizon extraction: the integration spec lives in the BNGL simulate action,
# not in the _core manifest params (which carry only tier/source/sha256). We
# parse the representative ODE action's tokens, then resolve any non-numeric
# token (a parameter name or expression) against the .net's RESOLVED parameter
# table so models whose t_end/n_steps reference a parameter still run.
# --------------------------------------------------------------------------- #
# Matches every simulate-style action whose per-run integration we may want to
# time: simulate({…}) / simulate_<method>({…}), and the workflow actions
# parameter_scan / bifurcate (each runs a per-point simulation with its OWN
# method=>/t_end=> args — for our purposes a single representative ODE integration
# of the network). group(1)=action keyword, group(2)=simulate_ suffix method (or
# None), group(3)=the brace blob.
_SIM_BLOCK_RE = re.compile(
    r"(simulate(?:_(\w+))?|parameter_scan|bifurcate)\s*\(\s*\{([^}]*)\}", re.DOTALL
)


def _strip_comments(text: str) -> str:
    return re.sub(r"#.*", "", text)


def read_net_parameters(net_path: str | Path) -> dict[str, float]:
    """Parse ``begin parameters … end parameters`` from a BNG2.pl ``.net``.

    Lines are ``<index> <name> <value> [# comment]`` with the value already
    resolved to a float by BNG2.pl. Returns ``{name: value}`` for token
    resolution (a t_end/n_steps given as a parameter name or expression).
    """
    params: dict[str, float] = {}
    try:
        text = Path(net_path).read_text()
    except OSError:
        return params
    m = re.search(r"begin parameters(.*?)end parameters", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return params
    for line in m.group(1).splitlines():
        s = line.split("#", 1)[0].split()
        if len(s) >= 3:
            try:
                params[s[1]] = float(s[2])
            except ValueError:
                continue
    return params


def read_net_species_ics(net_path: str | Path) -> dict[int, str]:
    """Parameter-dependent seed-species initial concentrations from a ``.net`` (GH #181).

    Parses ``begin species … end species``; each line is
    ``<1-based index> <species-name> <conc-token>`` (a network species name carries no
    spaces, so the token is the third whitespace field). Returns ``{idx0: token}``
    (0-based, matching ``Model.species_names`` / ``get_state`` order) for every species
    whose ``conc-token`` is a PARAMETER reference (non-numeric) — a symbolic seed-species
    IC that BNG2.pl writes as a ``_InitialConc<N>`` / parameter name and RE-EVALUATES
    against the live parameter table on each ``writeNetwork``. Species with a numeric IC
    (and all generated complexes, IC 0) are omitted. The token resolves via
    ``Model.get_param`` — bngsim re-evaluates the full ConstantExpression chain after a
    ``set_param`` of any dependency (verified), so ``get_param(token)`` is the live IC.
    """
    ics: dict[int, str] = {}
    try:
        text = Path(net_path).read_text()
    except OSError:
        return ics
    m = re.search(r"begin species(.*?)end species", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return ics
    for line in m.group(1).splitlines():
        s = line.split("#", 1)[0].split()
        if len(s) < 3:
            continue
        try:
            idx = int(s[0])
        except ValueError:
            continue
        token = s[2]
        try:
            float(token)
        except ValueError:
            ics[idx - 1] = token  # non-numeric -> a parameter-dependent (symbolic) IC
    return ics


# Python keywords that are valid *literals* in an eval (True/False/None) — bngsim's
# _jacobian._LITERAL_KEYWORDS. A parameter so named is left as the literal (never
# aliased), matching the model block; the rest of keyword.kwlist breaks eval as an
# identifier and is aliased below.
_KW_LITERALS = frozenset({"True", "False", "None"})


def _resolve_token(token, params: dict[str, float]):
    """A simulate-arg token (literal, parameter name, or expression) → float|None.

    Tries a plain numeric eval first, then evaluates against the .net's resolved
    parameter table (so ``t_end=>tend`` or ``t_end=>10*tend`` resolve). Returns
    None when the token cannot be resolved to a finite number.

    GH #182: a BNGL parameter may be legally named with a Python keyword (e.g.
    ``lambda`` in ``(1+lambda)``). Raw ``eval`` parses it AS the keyword (SyntaxError)
    and the value is silently dropped. The model is fine — this evaluator just has to
    cope, so we reuse bngsim's OWN keyword-alias machinery (the same
    ``_PY_KEYWORD_PARAM_NAMES`` / ``_alias_keyword_param`` its codegen + Jacobian apply
    in the model block) to whole-word-substitute such params to a safe symbol in both
    the expression and the eval scope. Gated on a cheap stdlib check so the bngsim
    import is paid only when a keyword-named parameter is actually in scope.
    """
    if token is None:
        return None
    token = str(token).strip().strip("'\"")
    if not token:
        return None
    if any(keyword.iskeyword(p) for p in params):
        from bngsim._codegen import _PY_KEYWORD_PARAM_NAMES, _alias_keyword_param

        params = dict(params)
        kw = [p for p in params if p in _PY_KEYWORD_PARAM_NAMES and p not in _KW_LITERALS]
        for p in sorted(kw, key=len, reverse=True):  # longest first: e.g. alias before sub
            alias = _alias_keyword_param(p)
            token = re.sub(rf"\b{re.escape(p)}\b", alias, token)
            params[alias] = params[p]
    for env in ({}, params):
        try:
            val = eval(token, {"__builtins__": {}}, env)  # noqa: S307 — sandboxed env
            if isinstance(val, (int, float)) and np.isfinite(val):
                return float(val)
        except Exception:
            continue
    return None


# The method aliases each track accepts on a simulate action. ODE: the CVODE
# integrator (and its bare-``simulate`` default); SSA: the exact Gillespie binary;
# NF: the network-free engine. ``simulate_<m>`` suffixes (simulate_ssa/_nf) and an
# explicit ``method=>`` arg both resolve through these sets.
_ODE_METHODS = frozenset({"ode", "cvode"})
_SSA_METHODS = frozenset({"ssa"})
_NF_METHODS = frozenset({"nf", "nf_reject", "nf_exact"})


def parse_sim_spec(
    text: str,
    net_params: dict[str, float],
    *,
    methods,
    atol: float,
    rtol: float,
    default_method: str = "ode",
):
    """Representative simulate-action spec for one engine track, or None.

    Generalizes :func:`parse_ode_spec` across the deterministic/stochastic tracks:
    scans every ``simulate({method=>…})`` / ``simulate_<m>({…})`` action whose
    resolved method is in ``methods`` (e.g. ``_ODE_METHODS`` / ``_SSA_METHODS`` /
    ``_NF_METHODS``) and picks the one with the largest resolved ``(t_end-t_start)``
    span — the dominant simulation cost, the run worth timing. A bare ``simulate``
    (no method) and ``parameter_scan``/``bifurcate`` default to ``default_method``
    (per-point ODE), so the ODE track still matches them. Token values resolve
    against ``net_params`` (the .net's parameter table). ``atol``/``rtol`` are kept
    on the returned spec for the deterministic track; the stochastic tracks ignore
    them but carry them harmlessly.

    Returns ``{t_start, t_end, n_steps, atol, rtol}`` or None when the model has no
    matching action or its t_end cannot be resolved to a number.
    """

    def _arg(blob, key, default=None):
        am = re.search(rf"{key}\s*=>\s*([^,}}]+)", blob)
        return _resolve_token(am.group(1), net_params) if am else default

    text = _strip_comments(text)
    best = None  # (span, spec)
    for m in _SIM_BLOCK_RE.finditer(text):
        suffix_method, blob = m.group(2), m.group(3)
        # method: a simulate_<m> suffix wins; else the block's method=> arg; else the
        # track default (ode — the default for simulate and parameter_scan/bifurcate).
        if suffix_method:
            method = suffix_method
        else:
            mm = re.search(r"method\s*=>\s*['\"]?(\w+)['\"]?", blob)
            method = mm.group(1) if mm else default_method
        if method.lower() not in methods:
            continue

        t_end = _arg(blob, "t_end")
        if t_end is None:
            continue
        t_start = _arg(blob, "t_start", 0.0) or 0.0
        n_steps = _arg(blob, "n_steps") or _arg(blob, "n_output_steps")
        gml = _arg(blob, "gml")  # NF global molecule limit; None for ode/ssa actions
        spec = {
            "t_start": t_start,
            "t_end": t_end,
            "n_steps": int(n_steps) if n_steps and n_steps >= 1 else DEFAULT_N_STEPS,
            "atol": _arg(blob, "atol", atol) or atol,
            "rtol": _arg(blob, "rtol", rtol) or rtol,
            "gml": int(gml) if gml and gml >= 1 else None,
        }
        span = t_end - t_start
        if span > 0 and (best is None or span >= best[0]):
            best = (span, spec)
    return best[1] if best else None


def parse_ode_spec(text: str, net_params: dict[str, float], *, atol: float, rtol: float):
    """Representative ODE integration spec for a BNGL model, or None.

    Thin wrapper over :func:`parse_sim_spec` for the ODE track (CVODE actions +
    the bare-``simulate``/``parameter_scan``/``bifurcate`` ODE default). Returns
    ``{t_start, t_end, n_steps, atol, rtol}`` or None.
    """
    return parse_sim_spec(
        text, net_params, methods=_ODE_METHODS, atol=atol, rtol=rtol, default_method="ode"
    )


def parse_stoch_spec(text: str, net_params: dict[str, float], *, track: str):
    """Representative SSA/NF simulate spec for the stochastic ``track``, or None.

    ``track`` is ``"ssa"`` or ``"nf"``; selects the matching ``simulate_<m>`` /
    ``method=>`` actions (``_SSA_METHODS`` / ``_NF_METHODS``) and, for SSA, also a
    bare ``simulate`` only when its method explicitly resolves to ssa (default_method
    is set to the track itself so a method-less ``simulate`` on a single-action SSA
    model is still picked up). Tolerances are irrelevant to a stochastic run, so the
    shared defaults are passed and ignored. Returns ``{t_start, t_end, n_steps,
    atol, rtol}`` or None.
    """
    methods = _SSA_METHODS if track == "ssa" else _NF_METHODS
    return parse_sim_spec(
        text,
        net_params,
        methods=methods,
        atol=DEFAULT_ATOL,
        rtol=DEFAULT_RTOL,
        default_method=track,
    )


# --------------------------------------------------------------------------- #
# Network generation (BNG2.pl) — the shared per-model build prefix.
# --------------------------------------------------------------------------- #
# Action statements stripped before we append a generate_network so the netgen
# pass produces ONLY the network (no run_network/NFsim sim, no writeMfile abort).
_ACTION_LINE_RE = re.compile(
    r"^(generate_network|simulate\w*|parameter_scan|bifurcate|readFile|setOption|"
    r"writeSBML|writeMfile|writeModel|writeXML|writeFile|writeNET|writeNetwork|"
    r"writeMexfile|visualize|setConcentration|addConcentration|saveConcentrations|"
    r"saveParameters|resetConcentrations|resetParameters|setParameter|quit)\b"
)

# ...but a ``setOption`` that configures NETWORK GENERATION (not the simulation) must
# survive into the netgen body, or BNG2.pl generates the WRONG network (GH #176
# follow-up). ``setOption("NumberPerQuantityUnit", N)`` sets the bimolecular
# concentration→count unit conversion to ``1/(N·V)``; energy-BNG models (catalysis,
# mwc, wofsy_goldstein, energy_example1) set N=NA=6.0221e23 before ``begin model``.
# Stripping it made BNG2.pl default to N=1 ⇒ conversion ``1/V`` ⇒ bimolecular rate
# constants ~NA (6e23×) too large: catalysis became genuinely unintegrable (1e28
# fluxes) and the other three got silently-wrong (degenerate) trajectories, while
# canonical BNG2.pl (with the option) runs them all correctly. ``SpeciesLabel`` is
# NOT preserved — it changes only internal species canonical labels, not the
# dynamics or the user-named observables, so it stays stripped (status quo). A
# preserved ``setOption`` inside ``begin actions``/after ``end model`` is still
# dropped by the block strip / end-model truncation; only model-header options
# BNG2.pl must see at netgen survive.
_NETGEN_PRESERVE_SETOPTION_RE = re.compile(
    r'^\s*setOption\s*\(\s*"?NumberPerQuantityUnit"?', re.IGNORECASE
)


def _strip_action_lines(text: str) -> str:
    """Drop every action line (``_ACTION_LINE_RE``) except a netgen-affecting
    ``setOption`` (``_NETGEN_PRESERVE_SETOPTION_RE``), which BNG2.pl must see to
    generate the correct network."""
    return "\n".join(
        ln
        for ln in text.splitlines()
        if not _ACTION_LINE_RE.match(ln.strip()) or _NETGEN_PRESERVE_SETOPTION_RE.match(ln)
    )


# Top-level (un-blocked) actions follow `end model`; matching it lets netgen drop
# everything after the model definition, so NO action — even an action verb absent
# from _ACTION_LINE_RE — can survive to run before (and break) the generate_network
# we append (e.g. a stray writeNetwork/writeMexfile needs a network to exist first).
_END_MODEL_RE = re.compile(r"^\s*end\s+model\b.*$", re.IGNORECASE | re.MULTILINE)

_GEN_NETWORK_RE = re.compile(r"generate_network\s*\(\s*\{[^}]*\}\s*\)")


def injected_action_block(overrides) -> str | None:
    """The ``rehab``/``action_inject`` action block for a job, or None.

    Some corpus models ship without a runnable simulation protocol (a dud body
    plus no/aborting actions); the suite's overrides REPLACE their actions with a
    short fixture that exercises the engine. ``overrides`` is the job's
    ``_core.Override`` list (objects with ``.field``/``.value``). The fixture is the
    one authoritative source of this model's simulate horizon (and, for the SSA/ODE
    netgen, its network caps), so the direct-drive runner parses the horizon from
    it and feeds its ``generate_network`` to netgen — otherwise an action-less model
    has no resolvable horizon and is dropped as SKIP.
    """
    for ov in overrides or []:
        if getattr(ov, "field", None) in ("rehab", "action_inject"):
            return getattr(ov, "value", None)
    return None


def injected_gen_network(action_block: str | None) -> str | None:
    """The ``generate_network(...)`` call inside an injected action block, or None.

    A network-free NF fixture has none (writeXML emits the XML directly); an SSA/ODE
    fixture carries the ``max_iter``/``max_agg`` caps that bound an otherwise
    incomplete or explosive network, which netgen must honor for the two engines to
    agree on the same fixed network.
    """
    if not action_block:
        return None
    m = _GEN_NETWORK_RE.search(action_block)
    return m.group(0) if m else None


def _netgen_bngl(text: str, gen_network: str | None = None, state_prefix: str = "") -> str:
    """Strip every action block/line and append a generate_network.

    Netgen depends only on the model blocks (parameters/species/rules), so this
    yields the same network the model's own actions would, without running (and
    timing) a simulation inside BNG2.pl. The .net carries the resolved parameter
    table + groups block we need.

    ``gen_network`` overrides the appended call with a specific ``generate_network``
    (with its caps) — used for an injected-action job whose fixture sets the network
    caps (``max_iter``/``max_agg``) the model needs to expand to a complete, bounded
    network. ``None`` keeps the bare default, so every non-injected model's .net is
    byte-identical to before.

    ``state_prefix`` (see :func:`state_setup_prefix`, GH #177) is replayed BEFORE the
    appended ``generate_network`` so the .net's parameter table / seed-species amounts
    reflect the ``setParameter``/``setConcentration`` state that precedes the
    representative simulate — the ordering matters: state set after ``generate_network``
    is NOT baked into the .net. Empty keeps the artifact byte-identical to before.
    """
    text = re.sub(r"begin actions.*?end actions", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Drop everything after the model definition (top-level actions live there), so
    # an unlisted action verb can't survive to run before the appended netgen.
    m = list(_END_MODEL_RE.finditer(text))
    if m:
        text = text[: m[-1].end()]
    body = _strip_action_lines(text)
    pre = (state_prefix.rstrip() + "\n") if state_prefix.strip() else ""
    return body.rstrip() + "\n" + pre + (gen_network or "generate_network({overwrite=>1})") + "\n"


def generate_network(
    bngl_text: str,
    bng2_pl: str,
    workdir: Path,
    *,
    timeout: float,
    gen_network: str | None = None,
    state_prefix: str = "",
) -> tuple[Path | None, float, str]:
    """Run BNG2.pl on a netgen-only copy of ``bngl_text`` in ``workdir``.

    Returns ``(net_path, netgen_sec, error)``. ``net_path`` is None on failure
    (``error`` then carries the last stderr/stdout line). ``netgen_sec`` is the
    wall around the BNG2.pl subprocess — the shared per-model build prefix.
    ``gen_network`` (see :func:`_netgen_bngl`) overrides the appended call for an
    injected-action job; ``None`` keeps the bare default. ``state_prefix`` (GH #177)
    replays the pre-simulate ``setParameter``/``setConcentration`` state into the .net.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    net_path = workdir / "model.net"
    # A model whose OWN ``generate_network(...)`` call (preserved for its caps by
    # _model_gen_network) lacks ``overwrite=>1`` makes BNG2.pl ABORT ("Previously
    # generated model.net exists") when re-swept into a workdir that still holds a
    # prior run's .net — surfacing as a truncated "...failed: at line N" on the
    # injected generate_network line. Clear the stale artifact so every re-sweep
    # regenerates from scratch regardless of the call's overwrite flag.
    net_path.unlink(missing_ok=True)
    bngl_path = workdir / "model.bngl"
    bngl_path.write_text(_netgen_bngl(bngl_text, gen_network, state_prefix))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            ["perl", bng2_pl, str(bngl_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workdir),
        )
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t0, f"BNG2.pl netgen timed out after {timeout}s"
    netgen_sec = time.perf_counter() - t0
    if proc.returncode != 0 or not net_path.exists():
        tail = (proc.stderr.strip() or proc.stdout.strip() or "").splitlines()
        return None, netgen_sec, f"BNG2.pl netgen failed: {(tail[-1] if tail else '')[:300]}"
    return net_path, netgen_sec, ""


def _writexml_bngl(text: str, state_prefix: str = "") -> str:
    """Strip every action block/line and append a bare ``writeXML()``.

    The network-free analogue of :func:`_netgen_bngl`. NFsim (and bngsim's NF path)
    consume BNG-XML, NOT a reaction-network ``.net`` — and ``generate_network``
    would explode on a network-free model anyway — so we emit XML directly from the
    rules. ``writeXML`` depends only on the model blocks (parameters/molecule
    types/species/rules/observables), so this yields the same XML the model's own
    actions would, without running (and timing) a simulation inside BNG2.pl.

    ``state_prefix`` (see :func:`state_setup_prefix`, GH #177) is replayed BEFORE
    ``writeXML()`` so the XML's parameter table / seed-species amounts reflect the
    pre-simulate ``setParameter``/``setConcentration`` state (NFsim resolves the
    species' symbolic concentrations against this table — e.g. ``setParameter("S0",1)``
    drops a 331e6 default population to 1). Empty keeps the XML byte-identical.
    """
    text = re.sub(r"begin actions.*?end actions", "", text, flags=re.DOTALL | re.IGNORECASE)
    m = list(_END_MODEL_RE.finditer(text))  # drop top-level actions after end model
    if m:
        text = text[: m[-1].end()]
    body = _strip_action_lines(text)
    pre = (state_prefix.rstrip() + "\n") if state_prefix.strip() else ""
    return body.rstrip() + "\n" + pre + "writeXML()\n"


def generate_xml(
    bngl_text: str, bng2_pl: str, workdir: Path, *, timeout: float, state_prefix: str = ""
) -> tuple[Path | None, float, str]:
    """Run BNG2.pl on a writeXML-only copy of ``bngl_text`` → BNG-XML in ``workdir``.

    The NF-track analogue of :func:`generate_network` (the shared per-model build
    *prefix* for the network-free track): the same ``.xml`` feeds BOTH the legacy
    ``NFsim`` binary and bngsim's in-process NF engine, so it is measured once and
    attributed to neither. Returns ``(xml_path, gen_sec, error)``; ``xml_path`` is
    None on failure (``error`` carries the last stderr/stdout line). BNG2.pl names
    the XML after the model file (``model.xml``) but falls back to the first
    ``*.xml`` produced if not. ``state_prefix`` (GH #177) replays the pre-simulate
    ``setParameter``/``setConcentration`` state into the emitted XML.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    # Clear any prior run's XML so a re-sweep can't return a stale model.xml as a
    # false success when this writeXML fails after the file already existed (the
    # network-free analogue of the generate_network stale-.net guard above).
    for stale in workdir.glob("*.xml"):
        stale.unlink(missing_ok=True)
    bngl_path = workdir / "model.bngl"
    bngl_path.write_text(_writexml_bngl(bngl_text, state_prefix))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            ["perl", bng2_pl, str(bngl_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workdir),
        )
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t0, f"BNG2.pl writeXML timed out after {timeout}s"
    gen_sec = time.perf_counter() - t0
    xml_path = workdir / "model.xml"
    if not xml_path.exists():
        cands = sorted(workdir.glob("*.xml"))
        xml_path = cands[0] if cands else xml_path
    if proc.returncode != 0 or not xml_path.exists():
        tail = (proc.stderr.strip() or proc.stdout.strip() or "").splitlines()
        return None, gen_sec, f"BNG2.pl writeXML failed: {(tail[-1] if tail else '')[:300]}"
    return xml_path, gen_sec, ""


# --------------------------------------------------------------------------- #
# .gdat / .cdat reader
# --------------------------------------------------------------------------- #
def read_dat(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read a BNG ``.cdat``/``.gdat`` → (time, values[n_time,n_var], names).

    The header is the leading ``#``-comment line of column names; data rows are
    whitespace-delimited floats with time in column 0.
    """
    names: list[str] | None = None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                names = s.lstrip("#").split()
            break
    data = np.loadtxt(str(path), comments="#", ndmin=2)
    col = (
        names[1:] if names and len(names) > 1 else [f"S{i + 1}" for i in range(data.shape[1] - 1)]
    )
    return data[:, 0].copy(), data[:, 1:], col


# --------------------------------------------------------------------------- #
# BNGsim adapter — in-process .net ODE with phase timing + cold/warm split.
# --------------------------------------------------------------------------- #
def bn_ode_net(
    net_path: str | Path,
    t_start: float,
    t_end: float,
    n_points: int,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """One BNGsim ODE run over a BNG2.pl ``.net``. Returns (time, values, names, timing).

    ``t_span=(t_start, t_end)`` sampled at ``n_points`` uniformly — set to
    ``n_steps + 1`` by the caller so the grid matches run_network's
    (step_size, n_steps) output exactly. The timing dict carries the bngsim build
    phases — ``load_sec`` (the ``from_net`` wall: C++ .net parse + interpret, the
    one phase a .net has no finer split for), ``jac_derive_sec``
    (``last_jacobian_sec``, analytical Functional Jacobian), ``codegen_sec``
    (``last_codegen_sec``, RHS backend build) — plus the cold→warm integration
    split (:func:`_integrate_stats`) and the resolved engine config. The one-time
    SymPy import is per-process warmup (measure_warmup), not charged here.
    """
    import bngsim

    t0 = time.perf_counter()
    model = bngsim.Model.from_net(str(net_path))
    load_sec = time.perf_counter() - t0

    sim = bngsim.Simulator(model, method="ode")

    # Cold solve (one-time CVODE setup + lazy Jacobian derivation + codegen +
    # solve) feeds the parity verdict; then up to _warm_rep_count(cold) warm
    # solves reuse the built model (reset() restores IC). warm-min = marginal
    # per-integration cost; cold = first-solve cost of a fresh Simulator.
    t1 = time.perf_counter()
    r = sim.run(t_span=(t_start, t_end), n_points=n_points, rtol=rtol, atol=atol)
    cold_sec = time.perf_counter() - t1
    warm: list[float] = []
    for _ in range(_warm_rep_count(cold_sec)):
        try:
            model.reset()
            t1 = time.perf_counter()
            sim.run(t_span=(t_start, t_end), n_points=n_points, rtol=rtol, atol=atol)
            warm.append(time.perf_counter() - t1)
        except Exception:
            break
    integ = _integrate_stats(cold_sec, warm)

    stats = r.solver_stats if hasattr(r, "solver_stats") else {}
    ls_code = (stats or {}).get("linear_solver", 0)
    config = {
        "codegen": sim.codegen_backend,
        "jacobian": sim.jacobian_strategy,
        "linear_solver": LINEAR_SOLVER_NAMES.get(ls_code, f"kind_{ls_code}"),
        "cached": sim.codegen_cache_hit,
    }
    timing = {
        "io_sec": 0.0,
        # .net has no separate libSBML parse / interpret split — from_net is a
        # single C++ load boundary, reported as load_sec (the per-model build
        # tier sums load + jac + codegen).
        "load_sec": round(load_sec, 6),
        "jac_derive_sec": round(float(sim.last_jacobian_sec), 6),
        "codegen_sec": round(float(sim.last_codegen_sec), 6),
        **integ,
        "config": config,
    }
    return np.asarray(r.time), np.asarray(r.species), list(r.species_names), timing


# --------------------------------------------------------------------------- #
# Legacy run_network adapter — direct binary on the SAME .net.
# --------------------------------------------------------------------------- #
_RN_INIT_RE = re.compile(r"Initialization took\s+([0-9.eE+\-]+)\s+CPU")
_RN_PROP_RE = re.compile(r"Propagation took\s+([0-9.eE+\-]+)\s+CPU")
_RN_PROG_RE = re.compile(r"Program times:\s+([0-9.eE+\-]+)\s+CPU s\s+([0-9.eE+\-]+)\s+clock")


def _parse_run_network_timing(stdout: str) -> dict:
    """Extract run_network's self-reported CPU/clock split from its stdout."""
    out: dict[str, float] = {}
    if m := _RN_INIT_RE.search(stdout):
        out["init_cpu_sec"] = float(m.group(1))
    if m := _RN_PROP_RE.search(stdout):
        out["propagation_cpu_sec"] = float(m.group(1))
    if m := _RN_PROG_RE.search(stdout):
        out["total_cpu_sec"] = float(m.group(1))
        out["total_clock_sec"] = float(m.group(2))
    return out


def run_network_ode(
    net_path: str | Path,
    run_network_bin: str,
    *,
    t_start: float,
    t_end: float,
    n_steps: int,
    rtol: float,
    atol: float,
    out_prefix: str,
    timeout: float,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Integrate ``net_path`` with the legacy run_network CVODE binary.

    Builds the same command BNG2.pl would (``-p cvode -a atol -r rtol [-i t_start]
    --cdat 1 --fdat 0 -g <net> <net> <step_size> <n_steps>``) and runs it up to a
    budget-capped number of fresh-process reps. Each call is inherently COLD (a new
    process: read .net + CVODE setup + integrate), so there is no warm-solver reuse
    — we report the per-call wall distribution (min as the headline, to denoise the
    OS-scheduler jitter of a CLI spawn) and run_network's OWN init/propagation CPU
    split. Returns (time, values[n_time,n_var], species_names, timing); species are
    the ``.cdat`` columns (network species order, matching bn_ode_net). Raises
    RuntimeError if the binary fails.
    """
    step_size = (t_end - t_start) / n_steps
    cmd = [run_network_bin, "-o", out_prefix, "-p", "cvode", "-a", repr(atol), "-r", repr(rtol)]
    if t_start != 0.0:
        cmd += ["-i", repr(t_start)]
    cmd += [
        "--cdat",
        "1",
        "--fdat",
        "0",
        "-g",
        str(net_path),
        str(net_path),
        repr(step_size),
        str(int(n_steps)),
    ]

    walls: list[float] = []
    self_timing: dict = {}
    # First (cold-equivalent) call always runs; reps are budget-capped like bngsim.
    reps = 1 + _warm_rep_count(0.0)  # placeholder; recomputed after the first call
    i = 0
    while i < reps:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        wall = time.perf_counter() - t0
        if proc.returncode != 0:
            tail = (proc.stderr.strip() or proc.stdout.strip() or "").splitlines()
            raise RuntimeError(f"run_network failed: {(tail[-1] if tail else '')[:300]}")
        walls.append(wall)
        if i == 0:
            self_timing = _parse_run_network_timing(proc.stdout)
            reps = 1 + _warm_rep_count(wall)  # fewer reps for a slow model
        i += 1

    cdat = Path(f"{out_prefix}.cdat")
    if not cdat.exists():
        raise RuntimeError("run_network produced no .cdat")
    time_arr, vals, names = read_dat(cdat)

    cold = walls[0]
    reps_rest = walls[1:]
    timing = {
        "io_sec": 0.0,
        **_integrate_stats(cold, reps_rest),
        # run_network's own internal CPU split (read .net + CVODE setup vs the
        # integration loop); coarse (printed to 0.01 CPU s) but engine-attested.
        **self_timing,
        "n_calls": len(walls),
        "config": {
            "codegen": "C (compiled run_network binary)",
            "jacobian": "FD (CVODE difference-quotient)",
            "linear_solver": "Dense (built-in LU)",
            "cached": None,
        },
    }
    return time_arr, vals, names, timing


# --------------------------------------------------------------------------- #
# Alignment — both engines integrate the SAME .net, so species are in identical
# network order; align by index with a length guard (a mismatch is a structural
# loader divergence, surfaced loudly).
# --------------------------------------------------------------------------- #
def align_net_species(
    bn_vals: np.ndarray,
    rn_vals: np.ndarray,
    bn_time: np.ndarray,
    rn_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return (bn_aligned, rn_aligned, n_common_species) on a shared grid, or None.

    Both arrays come from the one .net, so column i is the same species in each and
    the time grids (n_steps+1 uniform points over [t_start,t_end]) coincide. We
    truncate to the common species/time counts (defensive: a row/col count that
    differs by one from a boundary rounding still compares the overlap) and return
    None only when there is nothing to compare.
    """
    n_sp = min(bn_vals.shape[1], rn_vals.shape[1])
    n_t = min(bn_vals.shape[0], rn_vals.shape[0], bn_time.shape[0], rn_time.shape[0])
    if n_sp == 0 or n_t == 0:
        return None
    return bn_vals[:n_t, :n_sp], rn_vals[:n_t, :n_sp], n_sp


def align_observables_by_name(
    bn_names: list[str],
    bn_vals: np.ndarray,
    leg_names: list[str],
    leg_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Align two ensemble stacks on the shared OBSERVABLE names, or None.

    The stochastic analogue of :func:`align_net_species`, by name rather than by
    index: the SSA/NF comparison is over the ``.gdat`` observables (NF has no
    species at all), and although the legacy stack and bngsim emit the same
    observable names in the same order, we intersect by name to be robust to a
    reordering or a dropped column. ``bn_vals``/``leg_vals`` are
    ``(n_rep, n_time, n_obs)`` ensemble stacks. Returns
    ``(bn_aligned, leg_aligned, common_names)`` with both stacks restricted to the
    common observables in a shared order and truncated to the common time count, or
    None when the observable sets are disjoint.
    """
    bn_map = {n: i for i, n in enumerate(bn_names)}
    leg_map = {n: i for i, n in enumerate(leg_names)}
    common = sorted(set(bn_map) & set(leg_map))
    if not common:
        return None
    bi = [bn_map[n] for n in common]
    li = [leg_map[n] for n in common]
    bn_vals = np.asarray(bn_vals)
    leg_vals = np.asarray(leg_vals)
    n_t = min(bn_vals.shape[1], leg_vals.shape[1])
    if n_t == 0:
        return None
    return bn_vals[:, :n_t][:, :, bi], leg_vals[:, :n_t][:, :, li], common


# --------------------------------------------------------------------------- #
# Stochastic adapters (SSA / NF) — N-replicate ensembles, compared on the
# observables by name. The taxonomy mirrors the ODE adapters but to the
# stochastic cost model: a per-model load + a per-replicate ENSEMBLE (independent
# reseeded trajectories — no warm-reuse of one solve), plus the bngsim-only
# per-replicate setup. Both engines run the SAME shared artifact (the .net for
# SSA, the BNG-XML for NF), so the comparison is apples-to-apples.
# --------------------------------------------------------------------------- #
def bn_ssa_net(
    net_path: str | Path,
    t_start: float,
    t_end: float,
    n_points: int,
    n_rep: int,
    seed_base: int,
    *,
    rep_timeout: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """``n_rep`` BNGsim SSA replicates over a BNG2.pl ``.net``. Returns
    ``(time, obs[n_rep,n_time,n_obs], obs_names, timing)``.

    The model is loaded once (``Model.from_net`` — the .net's single C++ load
    boundary, timed as ``load_sec`` exactly like :func:`bn_ode_net`; NO Jacobian),
    the SSA Simulator is built once, and each replicate is ``model.reset()`` +
    ``sim.run(seed=seed_base+rep)`` so the legacy comparison uses an identical seed
    schedule. Eligible mass-action models compile a structure-specialized propensity
    vector on the first replicate (GH #190), reported in ``config.codegen``; the cold
    compile lands in the first (``rep_cold``) replicate, the warm steady-state in
    ``rep_median``. The returned values are the ``.gdat`` OBSERVABLES (by name) — the
    uniform stochastic comparison axis. ``rep_timeout`` bounds each trajectory (a
    runtime sign-indefinite SSA model can fire unboundedly). The per-replicate
    ``SsaBoundaryWarning`` is silenced (N× noise; the timing track does not adjudicate
    sign-indefiniteness — the correctness suite does).

    The ``timing`` dict carries the per-model SSA load, the per-replicate ensemble
    stats (:func:`_ensemble_stats` — ``sim.run`` only), the realized propensity
    backend (``config.codegen`` = cc / mir / interpreted), and the stochastic ACTIVITY
    (mean Gillespie events per replicate / per unit time — the real cost driver).
    """
    import statistics as _st
    import warnings

    import bngsim

    t0 = time.perf_counter()
    model = bngsim.Model.from_net(str(net_path))
    load_sec = time.perf_counter() - t0

    times: np.ndarray | None = None
    names: list[str] = []
    out: list[np.ndarray] = []
    run_secs: list[float] = []
    event_counts: list[int] = []
    prop_backend = "interpreted"
    # Build the Simulator ONCE, then reset()+run(seed=...) per replicate — the
    # build-once pattern run_network uses and the rr_parity SSA screen adopted (GH
    # #190). A per-replicate clone+construct is a tiny syscall/alloc-heavy op whose
    # wall, timed under the parallel sweep, is CPU-contention-dominated and showed up
    # as a confusing "setup/rep" that exceeded the trajectory cost. reset() restores
    # the initial state and run(seed=) seeds a fresh RNG, so the ensemble is
    # bit-identical to clone-per-rep on the same seed schedule.
    sim = bngsim.Simulator(model, method="ssa")
    for rep in range(n_rep):
        model.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", bngsim.SsaBoundaryWarning)
            t_run = time.perf_counter()
            r = sim.run(
                t_span=(t_start, t_end),
                n_points=n_points,
                seed=seed_base + rep,
                timeout=rep_timeout,
            )
            run_secs.append(time.perf_counter() - t_run)
        if times is None:
            times = np.asarray(r.time)
            names = list(r.observable_names)
            prop_backend = (getattr(r, "ssa_diagnostics", None) or {}).get(
                "propensity_backend", "interpreted"
            ) or "interpreted"
        out.append(np.asarray(r.observables))
        event_counts.append(int((r.solver_stats or {}).get("n_steps", 0) or 0))

    span = max(float(t_end) - float(t_start), 1e-12)
    mean_events = _st.mean(event_counts) if event_counts else 0.0
    # Report the propensity backend the engine ACTUALLY used (GH #190): structure-
    # specialized codegen is the default for eligible mass-action models, so the old
    # hardcoded "ExprTk (no codegen)" label was stale (it mislabeled cc-compiled runs).
    _rhs_label = {
        "cc": "Native C propensity vector (cc-compiled .so) — recompute-all",
        "mir": "MIR-JIT propensity vector — recompute-all",
        "interpreted": "ExprTk propensities (interpreted)",
    }.get(prop_backend, prop_backend)
    timing = {
        "io_sec": 0.0,
        # .net load has no separate libSBML/interpret split — from_net is one C++
        # boundary, reported as load_sec (the per-model build tier), matching
        # bn_ode_net. The .last_* accessors are ~0 for a .net and kept for provenance.
        "parse_sec": round(float(getattr(model, "last_libsbml_parse_sec", 0.0)), 6),
        "interpret_sec": round(float(getattr(model, "last_interpret_sec", 0.0)), 6),
        "load_sec": round(load_sec, 6),
        **_ensemble_stats(run_secs),
        "events_per_rep": round(mean_events, 3),
        "events_per_time": round(mean_events / span, 4),
        "events_total": int(sum(event_counts)),
        "config": {"method": "Gillespie SSA (exact)", "rhs": _rhs_label, "codegen": prop_backend},
    }
    return times, np.stack(out, axis=0), names, timing


def bn_nf_xml(
    xml_path: str | Path,
    t_start: float,
    t_end: float,
    n_points: int,
    n_rep: int,
    seed_base: int,
    *,
    block_same_complex_binding: bool = True,
    molecule_limit: int | None = None,
    rep_timeout: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """``n_rep`` BNGsim network-free replicates over a BNG-XML. Returns
    ``(time, obs[n_rep,n_time,n_obs], obs_names, timing)``.

    Uses the stateful ``NfsimSession`` (the NF path needs only the XML — no ``.net``,
    no species): the session is built once (XML parse + NFsim system construction,
    timed as ``load_sec``), then ``initialize(seed)`` + ``simulate`` run each
    replicate. Re-initializing one session with a fresh seed resets state to the
    initial population (verified), so the build cost is paid once and the
    per-replicate cost is just initialize+simulate — the ``setup_per_rep_sec``
    (initialize) is split out of the per-trajectory headline, mirroring the SSA
    clone+construct. ``block_same_complex_binding`` defaults to bngsim's correct
    ``True``; the caller may set it False to match the legacy NFsim binary's
    same-complex-binding semantics for an apples-to-apples timing comparison.
    Returns the observables by name (network-free → there are no species).
    """
    import statistics as _st

    import bngsim

    t0 = time.perf_counter()
    sess = bngsim.NfsimSession(
        str(xml_path),
        block_same_complex_binding=block_same_complex_binding,
        molecule_limit=molecule_limit,
    )
    load_sec = time.perf_counter() - t0

    times: np.ndarray | None = None
    names: list[str] = []
    out: list[np.ndarray] = []
    setup_secs: list[float] = []
    run_secs: list[float] = []
    reaction_counts: list[int] = []
    try:
        for rep in range(n_rep):
            t_setup = time.perf_counter()
            sess.initialize(seed=seed_base + rep)
            setup_secs.append(time.perf_counter() - t_setup)
            t_run = time.perf_counter()
            r = sess.simulate(t_start, t_end, n_points, timeout=rep_timeout)
            run_secs.append(time.perf_counter() - t_run)
            if times is None:
                times = np.asarray(r.time)
                names = list(r.observable_names)
            out.append(np.asarray(r.observables))
            # NFsim's globalEventCounter (one tick per reaction firing) surfaces on
            # the per-replicate Result as solver_stats["n_steps"] — the exact NF
            # mirror of the SSA total_steps chain (bn_ssa_net). initialize(seed)
            # rebuilds the System, so the counter is per-replicate, not cumulative.
            reaction_counts.append(int((r.solver_stats or {}).get("n_steps", 0) or 0))
    finally:
        sess.destroy()

    span = max(float(t_end) - float(t_start), 1e-12)
    mean_reactions = _st.mean(reaction_counts) if reaction_counts else 0.0
    total_reactions = int(sum(reaction_counts))
    total_run_sec = sum(run_secs)
    timing = {
        "io_sec": 0.0,
        # NfsimSession construction = the XML parse + NFsim system build; the
        # network-free analogue of the SSA .net load (no Jacobian/codegen).
        "load_sec": round(load_sec, 6),
        **_ensemble_stats(run_secs),
        "setup_per_rep_sec": round(_st.mean(setup_secs), 6) if setup_secs else None,
        # NF reaction-firing ENGINE throughput — the harness-stripped per-trajectory
        # metric. reactions_per_sec = total firings / total sim-wall (the sess.simulate
        # time only, excluding load+setup) so it is directly comparable to NFsim's own
        # self-reported reactions/sec. reactions_per_time mirrors SSA's events_per_time.
        "reactions_per_rep": round(mean_reactions, 3),
        "reactions_per_time": round(mean_reactions / span, 4),
        "reactions_total": total_reactions,
        "reactions_per_sec": round(total_reactions / total_run_sec, 2)
        if total_run_sec > 0
        else None,
        "config": {
            "method": "NFsim-style network-free (nf_reject)",
            "rhs": "rule-based, no network",
            "block_same_complex_binding": bool(block_same_complex_binding),
        },
    }
    return times, np.stack(out, axis=0), names, timing


# --------------------------------------------------------------------------- #
# Genuine-bngsim per-job driver (GH #175).
#
# The honest replacement for ``bionetgen.run(<bngl>, simulator='bngsim')``. With a
# STOCK BNG2.pl, the merged PyBioNetGen bridge routes a BNGL ``simulate`` to the
# BNG2.pl-owned backend hook (``ROUTE_BNGL_BNGSIM``); stock BNG2.pl carries no such
# hook, so it runs ``run_network`` / NFsim natively — bngsim never executes, yet
# the output is labelled "bngsim". So a ``simulator='bngsim'`` sweep silently
# produced BNG2.pl output (the golden was BNG2.pl-vs-bngsim mislabelled bngsim).
#
# Here bngsim ACTUALLY runs: BNG2.pl is used ONLY to GENERATE the reaction network
# (``.net``) or BNG-XML (``.xml``) — the one thing bngsim cannot do (it has no BNGL
# parser) — then the model is simulated IN-PROCESS through the bridge's DIRECT
# route (``ROUTE_DIRECT_BNGSIM`` -> ``execute_bngsim_direct_job``), the same path a
# downstream consumer takes for a ``.net``/``.xml`` artifact. That writes the
# consumer-faithful ``.gdat`` (observables) + ``.cdat`` (species) and is PROVABLY
# bngsim: with ``run_network`` removed it still succeeds, while the BNGL bridge
# route crashes (see test_bngsim_golden_engine.py). bngsim supports ode / ssa /
# psa (network) and nf / rm (network-free); only ``pla`` is unimplemented and a
# pla-only job is skipped rather than silently run on the legacy stack.
# --------------------------------------------------------------------------- #

# Stochastic methods bngsim implements. SSA + population SSA (psa) run on the
# ``.net``; NFsim (nf*) + RuleMonkey (rm) are network-free and run on the BNG-XML.
_PSA_METHODS = frozenset({"psa"})
_RM_METHODS = frozenset({"rm"})
# Track -> the resolved simulate-action methods that select it. ``psa`` also
# accepts a bare ``ssa`` action since BNG2.pl auto-promotes ssa+poplevel to psa.
# The two network-free tracks are method-faithful: nf -> NFsim, rm -> RuleMonkey.
_TRACK_METHODS = {
    "ode": _ODE_METHODS,
    "ssa": _SSA_METHODS,
    "psa": _PSA_METHODS | _SSA_METHODS,
    "nf": _NF_METHODS,
    "rm": _RM_METHODS,
}
_NETFREE_TRACKS = frozenset({"nf", "rm"})  # consume BNG-XML, not a .net
_POPLEVEL_RE = re.compile(r"poplevel\s*=>\s*([0-9.eE+\-]+)")
_SEED_RE = re.compile(r"seed\s*=>\s*(\d+)")


def _is_stochastic_text(text: str) -> bool:
    """True if any active simulate-style action uses a non-ODE method (ssa/nf/...).

    The same regime test the sweep applies (a non-``ode``/``cvode`` method makes
    the model stochastic), kept self-contained so :func:`run_bngsim_job` can decide
    the track family without importing the sweep.
    """
    return any(m not in _ODE_METHODS for m in _simulate_methods(text))


def _parse_seed(text: str) -> int | None:
    """The baked-in ``seed=>K`` the sweep injected into the patched model, or None."""
    m = _SEED_RE.search(_strip_comments(text))
    return int(m.group(1)) if m else None


def resolve_bng2_pl(bngpath: str | None) -> str:
    """Resolve BNG2.pl from a ``$BNGPATH``-style value (a dir OR a direct file)."""
    if not bngpath:
        raise RuntimeError(
            "BNGPATH/BNG2_PL is unset; bngsim needs BNG2.pl to generate the network"
        )
    p = Path(bngpath)
    if p.is_dir():
        cand = p / "BNG2.pl"
        if cand.exists():
            return str(cand)
        raise RuntimeError(f"no BNG2.pl in directory {p}")
    if not p.exists():
        raise RuntimeError(f"BNG2.pl path does not exist: {p}")
    return str(p)


def _simulate_methods(text: str) -> set[str]:
    """Resolved method of every simulate-style action in ``text`` (lowercased).

    A ``ssa`` action carrying ``poplevel`` is reported as ``psa`` (BNG2.pl
    auto-promotes it and bngsim runs population SSA), matching the bridge's own
    routing so the track picker below agrees with the engine that actually runs.
    """
    out: set[str] = set()
    for m in _SIM_BLOCK_RE.finditer(_strip_comments(text)):
        suffix_method, blob = m.group(2), m.group(3)
        if suffix_method:
            method = suffix_method.lower()
        else:
            mm = re.search(r"method\s*=>\s*['\"]?(\w+)['\"]?", blob)
            method = mm.group(1).lower() if mm else "ode"
        if method in _SSA_METHODS and "poplevel" in blob:
            method = "psa"
        out.add(method)
    return out


def classify_bngsim_track(text: str, *, stochastic: bool) -> str | None:
    """Engine track for the genuine-bngsim drive: 'ode'|'ssa'|'psa'|'nf'|'rm'|None.

    Routing is METHOD-FAITHFUL — the model gets the engine it asks for:

      * ``method=>"nf"`` (nf/nf_reject/nf_exact) -> **NFsim** (``nf`` track),
      * ``method=>"rm"`` -> **RuleMonkey** (``rm`` track),
      * population SSA (``psa``), then exact SSA (``ssa``), then ODE.

    NFsim and RuleMonkey are both vendored into bngsim and are intended to be
    swappable network-free engines (switch the method and the other engine runs the
    same model — RuleMonkey is still catching up on a few NFsim features, e.g.
    FunctionProduct rate laws, see richardposner/RuleMonkey#19). We never reroute
    one to the other: ``run_bngsim_job`` hard-ERRORS if the requested engine is
    missing from the build (GH #175 — no silent substitution).

    Returns None when the unit declares only methods bngsim cannot run (``pla``):
    the caller records a skip rather than silently falling back to the legacy
    stack and mislabelling it bngsim.
    """
    if not stochastic:
        return "ode"
    methods = _simulate_methods(text)
    if methods & _NF_METHODS:
        return "nf"  # method=>"nf" -> NFsim
    if methods & _RM_METHODS:
        return "rm"  # method=>"rm" -> RuleMonkey
    if methods & _PSA_METHODS:
        return "psa"
    if methods & _SSA_METHODS:
        return "ssa"
    # No stochastic ENGINE requested. A model flagged stochastic only by a
    # workflow label (a ``parameter_scan({method=>"protocol"})`` whose per-point
    # runs are ODE — ``protocol`` is not a simulation engine) still has a genuine
    # ODE integration to fingerprint, so fall back to the ODE track. Only a method
    # bngsim has NO engine for (``pla``) with nothing else left -> None (skip).
    if (methods & _ODE_METHODS) or (methods - {"pla"}):
        return "ode"
    return None


def _model_gen_network(text: str) -> str | None:
    """The model's OWN ``generate_network(...)`` call (with its caps), or None.

    Netgen strips all actions and appends a bare ``generate_network`` by default;
    a capped model needs ITS call preserved so the network expands to the same
    bounded set the model intends. Crucially this must keep ``max_stoich=>{...}`` —
    a NESTED-brace argument that a ``\\{[^}]*\\}`` regex truncates (dropping the cap
    makes an otherwise-bounded network explode and time out). So we scan to the
    paren that balances ``generate_network(`` rather than regex-matching one brace.
    ``None`` keeps the bare default (every uncapped model's .net is unchanged).
    """
    s = _strip_comments(text)
    idx = s.find("generate_network")
    while idx != -1:
        p = s.find("(", idx)
        if p != -1:
            depth = 0
            for j in range(p, len(s)):
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                    if depth == 0:
                        return s[idx : j + 1]
        idx = s.find("generate_network", idx + 1)
    return None


def _poplevel(text: str) -> float:
    """The model's ``poplevel`` (population-SSA threshold), or BNG2.pl's default 100."""
    m = _POPLEVEL_RE.search(_strip_comments(text))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 100.0


# --------------------------------------------------------------------------- #
# Pre-simulate state-setup replay (GH #177).
#
# netgen/writeXML strip EVERY action so BNG2.pl runs no simulation. But a model's
# ``setParameter``/``setConcentration``/``setOption`` actions that PRECEDE the
# simulate we reproduce are state SETUP, not simulation — dropping them runs the
# representative against the wrong initial state (silently wrong, or — when the
# stripped setup was what kept the run tractable, e.g. ``setParameter("S0",1)`` to
# replace a 331e6 default population — catastrophic). BNG2.pl bakes these actions
# into the emitted .net/.xml WITHOUT running a sim, so we keep the ones that run
# before the representative simulate and replay them through netgen.
# --------------------------------------------------------------------------- #
# Actions that only set state (params / seed amounts / options / save-reset) —
# safe and necessary to replay through netgen.
_STATE_ACTION_RE = re.compile(
    r"^(setParameter|setConcentration|addConcentration|setOption|"
    r"saveConcentrations|resetConcentrations|saveParameters|resetParameters)\b"
)
# Simulation/workflow actions — the ones netgen must NOT run (bngsim runs the
# representative). They also delimit the prefix of state we keep.
_SIM_ACTION_RE = re.compile(r"^(simulate\w*|parameter_scan|bifurcate)\b")
# A LABELED save/reset slot (``saveConcentrations("pre_scan")``): restores a named
# checkpoint, which the single-slot erasure tracker below cannot model — its presence
# forces ``dirty_carryover`` so the case is surfaced for full-replay review, never
# silently mis-read as a clean single-phase prefix.
_LABELED_SAVERESET_RE = re.compile(r"^(saveConcentrations|resetConcentrations)\s*\(\s*['\"]")
_ACTIONS_BLOCK_RE = re.compile(r"begin\s+actions(.*?)end\s+actions", re.DOTALL | re.IGNORECASE)


def _actions_region(text: str) -> str:
    """The model's action statements — ``begin actions`` block(s) + top-level tail.

    Actions live in a ``begin actions … end actions`` block and/or as top-level
    statements after ``end model``; collect both so the state-setup scan sees the
    same statements BNG2.pl would execute, in document order.
    """
    region = "\n".join(_ACTIONS_BLOCK_RE.findall(text))
    em = list(_END_MODEL_RE.finditer(text))
    if em:  # top-level actions after the last `end model`, minus any block already taken
        region += "\n" + _ACTIONS_BLOCK_RE.sub("", text[em[-1].end() :])
    return region


def _action_statements(region: str) -> list[str]:
    """One logical action statement per element, in order (comments/blanks dropped).

    Joins BNGL line-continuations (a trailing ``\\``) so a multi-line
    ``simulate({…\\ …})`` is a single statement.
    """
    region = re.sub(r"\\\s*\n", " ", _strip_comments(region))  # join continuations
    return [s for s in (ln.strip() for ln in region.splitlines()) if s]


def _representative_action_index(statements, methods, default_method) -> int | None:
    """Index in ``statements`` of the representative simulate for a track, or None.

    Mirrors :func:`parse_sim_spec`'s selection (largest resolved span, ties → last)
    so the kept state-setup prefix ends exactly where that run begins. Spans resolve
    NUMERICALLY — the .net parameter table is not available yet at netgen time; when
    no matching simulate has a numeric span, falls back to the LAST matching action
    so the whole state-setup chain is kept rather than truncated mid-protocol.
    """
    best = None  # (span, idx)
    last_match = None
    for i, s in enumerate(statements):
        m = _SIM_BLOCK_RE.match(s)
        if not m:
            continue
        suffix_method, blob = m.group(2), m.group(3)
        if suffix_method:
            method = suffix_method
        else:
            mm = re.search(r"method\s*=>\s*['\"]?(\w+)['\"]?", blob)
            method = mm.group(1) if mm else default_method
        if method.lower() not in methods:
            continue
        last_match = i
        am = re.search(r"t_end\s*=>\s*([^,}]+)", blob)
        t_end = _resolve_token(am.group(1) if am else None, {})
        if t_end is None:
            continue
        sm = re.search(r"t_start\s*=>\s*([^,}]+)", blob)
        t_start = _resolve_token(sm.group(1) if sm else None, {}) or 0.0
        span = t_end - t_start
        if span > 0 and (best is None or span >= best[0]):
            best = (span, i)
    return best[1] if best else last_match


def state_setup_prefix(text: str, *, track: str) -> tuple[str, dict]:
    """BNGL state-setup actions preceding ``track``'s representative simulate (GH #177).

    Returns ``(prefix, info)``. ``prefix`` is the newline-joined
    ``setParameter``/``setConcentration``/``setOption``/save-reset lines that run
    BEFORE the representative simulate — replayed through netgen/writeXML so BNG2.pl
    bakes that state into the .net/.xml WITHOUT running a sim. Simulation/workflow
    and write/visualize actions are dropped (bngsim runs the representative; netgen
    must run no sim). An empty prefix leaves the emitted artifact byte-identical to
    the strip-everything default, so unaffected models are unchanged.

    ``info`` carries ``representative_found``, ``kept`` (line count), ``dropped_sims``
    (intervening simulates dropped from the prefix), and ``dirty_carryover`` — True
    when a dropped simulate's concentration state is NOT erased by a later
    ``resetConcentrations`` before the representative, i.e. the run's initial state
    depends on a prior simulate's RESULT (full multi-phase protocol replay, not
    handled here; the prefix is still applied best-effort and the flag lets the audit
    surface it). Both known catastrophic cases (BSA_v10, scaling_example) are clean
    (``dirty_carryover`` False).
    """
    methods = _TRACK_METHODS.get(track, _ODE_METHODS)
    statements = _action_statements(_actions_region(text))
    rep = _representative_action_index(statements, methods, track)
    info = {
        "representative_found": rep is not None,
        "kept": 0,
        "dropped_sims": 0,
        "dirty_carryover": False,
    }
    if rep is None:
        return "", info
    kept: list[str] = []
    dirty = False  # a dropped simulate perturbed concentrations since the last clean reset
    saved_dirty = None  # dirtiness captured at the last saveConcentrations (None = never saved)
    labeled = False  # a labeled save/reset slot we cannot model -> force the flag
    for s in statements[:rep]:
        if _SIM_ACTION_RE.match(s):
            dirty = True
            info["dropped_sims"] += 1
        elif _STATE_ACTION_RE.match(s):
            if _LABELED_SAVERESET_RE.match(s):
                labeled = True  # named checkpoint — not single-slot modelable
            elif s.startswith("saveConcentrations"):
                saved_dirty = dirty
            elif s.startswith("resetConcentrations"):
                dirty = bool(saved_dirty)  # restore saved state (declared-clean if never saved)
            kept.append(s)
        # else: generate_network / write* / visualize / readFile / quit -> dropped
    info["kept"] = len(kept)
    info["dirty_carryover"] = dirty or labeled
    return "\n".join(kept), info


# --------------------------------------------------------------------------- #
# Full multi-phase protocol replay (GH #179).
#
# The option-1 state_setup_prefix above bakes the pre-simulate setParameter/
# setConcentration state into the .net, but it CANNOT reproduce a representative
# simulate whose initial state is a PRIOR simulate's end state (BNG2.pl carries
# concentrations across simulates by default; ``continue=>1`` governs only the
# time/output axis, not state). Those models carry ``dirty_carryover=True``.
#
# Option-2 drives bngsim IN-PROCESS through the ordered action protocol —
# simulate -> setParameter -> simulate -> … — carrying each segment's end state
# forward (``Model.set_state``; the Simulator integrates from the model's live
# state, it does NOT reset to the .net IC), applying setParameter/setConcentration
# to the live model between segments, and capturing the representative segment's
# trajectory. Validated against native BNG2.pl to integration tolerance for both
# setConcentration- and setParameter-perturbed multi-phase protocols.
# --------------------------------------------------------------------------- #
# State-setup action value parsers (the live-model mutations replayed in-process).
# The species/parameter NAME is quote-delimited, so an internal comma or paren in a
# pattern (``"R(Y1~0,Y2~0,dim)"``) is captured intact; the value runs to the final
# ``)`` and resolves (literal or parameter expression) against the .net table.
_SETPARAM_VALUE_RE = re.compile(r"setParameter\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*(.+)\)\s*;?\s*$")
_SETCONC_VALUE_RE = re.compile(
    r"(set|add)Concentration\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*(.+)\)\s*;?\s*$"
)
_SAVERESET_LABEL_VALUE_RE = re.compile(r"(save|reset)Concentrations\s*\(\s*['\"]([^'\"]*)['\"]")


def parse_protocol(text: str, *, methods, default_method: str, net_params: dict[str, float]):
    """Typed, ordered replay steps for a model's full action protocol (GH #179).

    Walks the model's action statements (``begin actions`` block + top-level tail, in
    document order) and emits one typed step per replayable action, up to AND INCLUDING
    the representative simulate for ``methods``. Step kinds:

      * ``sim`` — a simulate segment: ``{method, t_start, t_end, n_steps, atol, rtol,
        gml, continue}``. ``t_start`` is CONTINUATION-AWARE (GH #179 secondary bug):
        an explicit ``t_start=>`` wins; else a ``continue=>1`` segment starts at the
        PREVIOUS simulate's ``t_end`` (not 0); else 0.
      * ``setparam`` / ``setconc`` / ``addconc`` — a live-model mutation
        ``{name, value}`` (value resolved against ``net_params``).
      * ``save`` / ``reset`` — a ``saveConcentrations`` / ``resetConcentrations``
        checkpoint ``{label}`` (label None for the bare single slot).

    Other actions (setOption / write* / visualize / readFile / generate_network / quit)
    are not live-model state and are dropped. Returns ``(steps, rep_index)`` where
    ``rep_index`` indexes the representative simulate in ``steps`` — the matching-method
    segment with the largest continuation-aware ``(t_end - t_start)`` span (ties → last,
    matching :func:`parse_sim_spec`), falling back to the last matching segment when no
    span resolves numerically. ``rep_index`` is None when no matching simulate exists.
    """

    def _arg(blob, key, default=None):
        am = re.search(rf"{key}\s*=>\s*([^,}}]+)", blob)
        return _resolve_token(am.group(1), net_params) if am else default

    statements = _action_statements(_actions_region(text))
    steps: list[dict] = []
    prev_t_end = 0.0
    for stmt_idx, s in enumerate(statements):
        sm = _SIM_BLOCK_RE.match(s)
        if sm:
            # GH #69 / #179: ``parameter_scan`` and ``bifurcate`` are parameter SWEEPS,
            # not single time-series simulate segments — each runs an independent
            # per-point simulation and writes scan output, never a single representative
            # ``m.cdat``/``m.gdat``. They must not be replayed as a protocol segment nor
            # chosen as the representative the matrix compares against the native oracle:
            # doing so compared two different segments (bngsim's replay vs whatever file
            # the native run happened to leave) and manufactured false DIFFs. Drop them
            # here exactly like the other non-replayable actions below.
            if not sm.group(1).startswith("simulate"):
                continue
            suffix_method, blob = sm.group(2), sm.group(3)
            if suffix_method:
                method = suffix_method.lower()
            else:
                mm = re.search(r"method\s*=>\s*['\"]?(\w+)['\"]?", blob)
                method = mm.group(1).lower() if mm else default_method
            if method in _SSA_METHODS and "poplevel" in blob:
                method = "psa"  # BNG2.pl auto-promotes ssa+poplevel
            t_end = _arg(blob, "t_end")
            cont = bool(_arg(blob, "continue"))
            t_start_explicit = _arg(blob, "t_start")
            if t_start_explicit is not None:
                t_start = t_start_explicit
            elif cont:
                t_start = prev_t_end  # continue: time axis follows the prior segment
            else:
                t_start = 0.0
            n_steps = _arg(blob, "n_steps") or _arg(blob, "n_output_steps")
            gml = _arg(blob, "gml")
            steps.append(
                {
                    "kind": "sim",
                    "method": method,
                    "t_start": float(t_start),
                    "t_end": t_end,
                    "n_steps": int(n_steps) if n_steps and n_steps >= 1 else DEFAULT_N_STEPS,
                    "atol": _arg(blob, "atol"),
                    "rtol": _arg(blob, "rtol"),
                    "gml": int(gml) if gml and gml >= 1 else None,
                    "continue": cont,
                    "stmt_idx": stmt_idx,
                }
            )
            if t_end is not None:
                prev_t_end = t_end
            continue
        pm = _SETPARAM_VALUE_RE.match(s)
        if pm:
            # Keep the RAW value token alongside the net_params pre-resolution: a
            # ``ConstantExpression`` parameter (e.g. ``Lig_tot = (lig_conc*Na)*Vec``) is
            # written symbolically in the .net so read_net_parameters skips it, but the
            # loaded bngsim Model knows its resolved value — multi_segment_replay
            # re-resolves the token against the live model scope (and raises if it can't,
            # never silently skipping a state mutation — the GH #177 lesson).
            steps.append(
                {
                    "kind": "setparam",
                    "name": pm.group(1),
                    "value": _resolve_token(pm.group(2), net_params),
                    "value_token": pm.group(2),
                    "stmt_idx": stmt_idx,
                }
            )
            continue
        cm = _SETCONC_VALUE_RE.match(s)
        if cm:
            steps.append(
                {
                    "kind": "addconc" if cm.group(1) == "add" else "setconc",
                    "name": cm.group(2),
                    "value": _resolve_token(cm.group(3), net_params),
                    "value_token": cm.group(3),
                    "stmt_idx": stmt_idx,
                }
            )
            continue
        if s.startswith("saveConcentrations"):
            lm = _SAVERESET_LABEL_VALUE_RE.match(s)
            steps.append(
                {
                    "kind": "save",
                    "label": (lm.group(2) if lm else None) or None,
                    "stmt_idx": stmt_idx,
                }
            )
            continue
        if s.startswith("resetConcentrations"):
            lm = _SAVERESET_LABEL_VALUE_RE.match(s)
            steps.append(
                {
                    "kind": "reset",
                    "label": (lm.group(2) if lm else None) or None,
                    "stmt_idx": stmt_idx,
                }
            )
            continue
        # setOption / saveParameters / resetParameters / write* / visualize / readFile /
        # generate_network / quit — not live-model state we replay; dropped.

    best = None
    rep_index = None
    last_match = None
    for i, st in enumerate(steps):
        if st["kind"] != "sim" or st["method"] not in methods:
            continue
        last_match = i
        if st["t_end"] is None:
            continue
        span = st["t_end"] - st["t_start"]
        if span > 0 and (best is None or span >= best):
            best, rep_index = span, i
    if rep_index is None:
        rep_index = last_match
    # Truncate to the representative segment — nothing after it affects its result.
    if rep_index is not None:
        steps = steps[: rep_index + 1]
    return steps, rep_index


# GH #182: bngsim ``set_concentration`` resolves a species by its EXACT canonical
# ``.net`` string, so a ``setConcentration`` pattern that differs only by component
# order (``BSA_THEM(DNP~hidden,DNP~exposed)`` vs canonical ``(DNP~exposed,DNP~hidden)``)
# or compartment syntax (``@cell:TP53(..)`` vs ``@cell::TP53(..)``) raises and forces
# the whole multi-phase replay onto the single-segment fallback. These helpers map such
# a pattern to the canonical name — but ONLY on an unambiguous, bond-free, concrete
# match, so a wrong/ambiguous pattern still falls back (never a silent substitution).
def _canonical_species_key(s: str):
    """Structural key for a CONCRETE, single-molecule, bond-free species string.

    Returns ``(compartment, molecule, sorted(components))`` so two strings that differ
    only by component order or ``@c:``/``@c::`` compartment syntax share a key. Returns
    ``None`` for anything we must not guess about: a complex/bonded species (``.``/``!``),
    or a malformed pattern (e.g. the unbalanced-paren ``IGF1(ds,hs,label~cold`` typo) —
    the regex requires a balanced ``Name(...)``.
    """
    s = s.strip()
    if not s or "." in s or "!" in s:
        return None
    comp = None
    m = re.match(r"^@(\w+)::?(.*)$", s)
    if m:
        comp, s = m.group(1), m.group(2)
    m = re.match(r"^(\w+)\((.*)\)$", s)
    if not m:
        return None
    name, body = m.group(1), m.group(2)
    comps = tuple(sorted(c.strip() for c in body.split(",") if c.strip())) if body.strip() else ()
    return (comp, name, comps)


def resolve_species_name(name: str, canonical_names) -> str:
    """Map a ``setConcentration`` pattern to the canonical ``.net`` species name (GH #182).

    Exact name -> unchanged. Otherwise, for a concrete bond-free species, the UNIQUE
    canonical species sharing its structural key. If there is no match, the match is
    ambiguous, or the pattern is non-concrete/malformed, returns ``name`` unchanged so
    the caller's exact-name ``set_concentration`` raises and the replay falls back —
    we never substitute a guessed species.
    """
    canonical = list(canonical_names)
    if name in set(canonical):
        return name
    key = _canonical_species_key(name)
    if key is None:
        return name
    matches = [c for c in canonical if _canonical_species_key(c) == key]
    return matches[0] if len(matches) == 1 else name


def multi_segment_replay(
    net_path: str | Path,
    steps: list[dict],
    rep_index: int,
    *,
    track: str,
    atol: float,
    rtol: float,
    seed: int | None,
    poplevel: float,
):
    """Drive bngsim in-process through a network protocol's segments (GH #179).

    Loads the ``.net`` ONCE at its declared initial condition, then replays ``steps``
    in order: a ``sim`` step integrates the current live state over ``(t_start, t_end)``
    and carries its end state forward (``Model.set_state`` — no ``reset``, so the next
    segment inherits it, exactly as BNG2.pl carries concentrations across simulates);
    ``setparam``/``setconc``/``addconc`` mutate the live model (validated to propagate
    to the reused Simulator with no rebuild); ``save``/``reset`` snapshot/restore the
    concentration vector (plus the symbolic-IC set) into a LABELED cache keyed by the
    action's ``saveConcentrations``/``resetConcentrations`` label (GH #186 — BNG2.pl's
    cache is per-label, not a single slot). One Simulator is reused across same-method
    segments and rebuilt only when a segment's method changes.

    A seed species whose initial concentration is a PARAMETER expression (the ``.net``
    writes it as a ``_InitialConc<N>`` / parameter reference; see
    :func:`read_net_species_ics`) is RE-EVALUATED against the live parameter scope before
    each simulate FOR AS LONG AS its concentration entry is still symbolic — exactly
    BNG2.pl's ``writeNetwork`` semantics, where a ``setParameter`` that changes such a
    parameter re-initializes the species (GH #181). An entry stops being symbolic once a
    simulate runs (BNG2.pl reads the ``.cdat`` end state back as numbers) or an explicit
    ``setConcentration`` overwrites it; ``saveConcentrations``/``resetConcentrations``
    snapshot and restore which entries are symbolic, so a ``reset`` to a pre-simulate
    snapshot makes them re-evaluable again. This is the carry-over-faithful fix for the
    #181 divergence (``kinetics_mb1n`` free antigen ``Ag(...) min(Agtot,Agmax)``).

    Returns ``(result, info)`` where ``result`` is the bngsim ``Result`` for the
    representative segment (``steps[rep_index]``) and ``info`` carries the replayed
    ``segments`` count, the representative ``rep_method``, and ``reinit_ics`` (number of
    seed species re-initialized from a parameter-expression IC across the protocol —
    nonzero only when #181 fired). Raises if a segment requests a network-free engine
    (deferred), or if no representative segment is captured. Network tracks only
    (``ode``/``ssa``/``psa``); NF/RM handled elsewhere.
    """
    import bngsim

    model = bngsim.Model.from_net(str(net_path))
    # The model's FULLY-resolved parameter scope — authoritative for a value token a
    # symbolic .net ConstantExpression hides from read_net_parameters. Kept current as
    # setParameter mutates primaries (a later expression referencing a changed primary
    # then resolves correctly).
    model_params = {n: model.get_param(n) for n in model.param_names}

    # GH #181: seed species whose IC is a parameter expression, by 0-based species index.
    # ``symbolic`` tracks which entries are CURRENTLY re-evaluable (BNG2.pl: the entry is
    # still an expression, not yet a number read back from a simulate); it mutates as the
    # protocol runs. ``net_ics`` is the pristine declared set, restored by an unsaved
    # resetConcentrations. Only tokens bngsim exposes as parameters are kept (defensive).
    _param_set = set(model.param_names)
    net_ics = {i: t for i, t in read_net_species_ics(net_path).items() if t in _param_set}
    symbolic = dict(net_ics)
    # The pristine declared seed-species state, captured before any mutation: what an
    # unsaved (DEFAULT-label) resetConcentrations restores. BNG2.pl falls back here to the
    # species' OWN ``Concentration`` expressions; ``symbolic`` is re-marked ``net_ics`` on
    # that path so the entries re-evaluate against the live parameter table.
    declared_state = np.asarray(model.get_state(), dtype=float).copy()
    # GH #186: BNG2.pl's saveConcentrations/resetConcentrations are a LABELED cache
    # (``Cache``: a dict keyed by an optional label, DEFAULT when unlabeled), NOT a single
    # slot. Each label stores BOTH the concentration vector AND the still-symbolic IC set
    # as of that save; a reset restores its OWN label. Collapsing every label into one slot
    # (the prior bug) made ``resetConcentrations("t=0")`` restore the most recent OTHER save
    # (``"start_competition"``) — scrambling the seed-species ICs (and their symbolic set)
    # in any multi-snapshot protocol (HarmonicOscillator/IGF1R: hot/cold IGF1 swapped onto
    # the IGF1R-dimer IC). bngsim's core resolver is correct; this was harness-only.
    saved_states: dict[str | None, tuple[np.ndarray, dict[int, str]]] = {}
    name_to_idx = {name: i for i, name in enumerate(model.species_names)}
    reinit_ids: set[int] = set()

    def _reeval_symbolic_ics():
        """Re-initialize still-symbolic seed-species ICs from the live parameter scope.

        Mirrors BNG2.pl's writeNetwork: each symbolic concentration entry is recomputed
        from its parameter expression (``Model.get_param`` re-evaluates the full
        ConstantExpression chain after a ``set_param``) right before the simulate that
        consumes it. By INDEX (``set_state``) so an exact species name is never required.
        """
        if not symbolic:
            return
        state = np.asarray(model.get_state(), dtype=float).copy()
        for idx, token in symbolic.items():
            state[idx] = float(model.get_param(token))
            reinit_ids.add(idx)
        model.set_state(state)

    def _value(st):
        """Resolve a setParameter/setConcentration value against the live model, or raise.

        Prefers the model-scope resolution (covers ConstantExpression params); falls back
        to the net_params pre-resolution; raises if neither yields a number — a state
        mutation must never be silently skipped (GH #177).
        """
        v = _resolve_token(st.get("value_token"), model_params)
        if v is None:
            v = st.get("value")
        if v is None:
            raise RuntimeError(
                f"{st['kind']} {st['name']!r}: value {st.get('value_token')!r} did not "
                f"resolve against the model parameter scope (refusing to skip silently)"
            )
        return float(v)

    sim = None
    sim_method = None
    captured = None
    n_segments = 0
    for i, st in enumerate(steps[: rep_index + 1]):
        kind = st["kind"]
        if kind == "sim":
            method = st["method"]
            if method in _NF_METHODS or method in _RM_METHODS:
                raise RuntimeError(
                    f"multi-segment replay does not support a network-free ({method}) "
                    f"segment inside a {track} protocol (deferred: NF/RM session carry-over)"
                )
            if sim is None or method != sim_method:
                init_kw = {"poplevel": poplevel} if method in _PSA_METHODS and poplevel else {}
                sim = bngsim.Simulator(model, method=method, **init_kw)
                sim_method = method
            run_kw: dict = {}
            if method in _ODE_METHODS:
                run_kw["rtol"] = float(st["rtol"] or rtol)
                run_kw["atol"] = float(st["atol"] or atol)
            else:  # ssa / psa — seeded for a reproducible golden trajectory
                run_kw["seed"] = int(seed if seed is not None else 1)
            _reeval_symbolic_ics()  # GH #181: re-init parameter-expression ICs (writeNetwork)
            r = sim.run(
                t_span=(st["t_start"], st["t_end"]), n_points=int(st["n_steps"]) + 1, **run_kw
            )
            # Carry the end state forward — the next segment integrates from here. After a
            # simulate every entry is a number (BNG2.pl reads the .cdat back), so nothing
            # is symbolic until a resetConcentrations restores a pre-simulate snapshot.
            model.set_state(np.asarray(r.species)[-1])
            symbolic.clear()
            n_segments += 1
            if i == rep_index:
                captured = r
        elif kind == "setparam":
            val = _value(st)
            model.set_param(st["name"], val)
            model_params[st["name"]] = val  # keep the resolution scope current
        elif kind == "setconc":
            # GH #182: canonicalize the pattern to the .net species name (component
            # reorder / @c:->@c:: compartment), unique-match-only else unchanged.
            cname = resolve_species_name(st["name"], model.species_names)
            idx = name_to_idx.get(cname)
            if idx is not None:
                symbolic.pop(idx, None)  # an explicit setConcentration makes this numeric
            model.set_concentration(cname, _value(st))
        elif kind == "addconc":
            cname = resolve_species_name(st["name"], model.species_names)  # GH #182
            idx = name_to_idx.get(cname)
            if idx is not None and idx in symbolic:
                # add to the RE-EVALUATED symbolic IC (BNG2.pl evaluates the entry first)
                base = float(model.get_param(symbolic.pop(idx)))
            else:
                base = model.get_concentration(cname)
            model.set_concentration(cname, base + _value(st))
        elif kind == "save":
            # GH #186: cache this label's concentration vector AND its still-symbolic IC set
            # (BNG2.pl's saveConcentrations writes a labeled cache entry, overwriting any
            # entry under the same label). set_state(get_state()) is the faithful restore.
            saved_states[st.get("label")] = (
                np.asarray(model.get_state(), dtype=float).copy(),
                dict(symbolic),
            )
        elif kind == "reset":
            snap = saved_states.get(st.get("label"))
            if snap is not None:
                saved_state, saved_symbolic = snap
                model.set_state(saved_state.copy())
                symbolic = dict(saved_symbolic)
            else:
                # No save under this label — restore the declared ICs (BNG2.pl falls back to
                # the species' own concentration expressions for an unsaved DEFAULT reset; a
                # named-but-never-saved reset is a BNG2.pl error not reached by valid corpus
                # protocols, where every reset has a matching prior save).
                model.set_state(declared_state.copy())
                symbolic = dict(net_ics)
    if captured is None:
        raise RuntimeError("multi-segment replay captured no representative segment")
    return captured, {
        "segments": n_segments,
        "rep_method": sim_method,
        "reinit_ics": len(reinit_ids),
    }


def multi_segment_replay_netfree(
    xml_path: str | Path,
    steps: list[dict],
    rep_index: int,
    *,
    track: str,
    seed: int | None,
    gml: int | None,
    timeout: float | None = None,
    block_same_complex_binding: bool = True,
):
    """Drive an NF/RM network-free session through a protocol's segments (GH #179).

    The network-free analogue of :func:`multi_segment_replay`. A network-free engine
    (NFsim for ``nf``, RuleMonkey for ``rm``) has no ``.net`` / species vector — it
    carries a live AGENT population. ``NfsimSession`` / ``RuleMonkeySession`` are
    stateful: consecutive ``simulate`` calls continue from the carried state (the
    ``t_start``/``t_end`` are output-grid LABELS — the window DURATION advances physical
    time), so the protocol is replayed by building the session ONCE, ``initialize``-ing
    once, then per segment applying the live mutations and simulating WITHOUT re-init:

      * ``setparam`` -> ``set_param`` (NFsim propagates to rates live; a post-init param
        change does NOT recreate the population — that is a network-free limitation we
        accept, not silently work around).
      * ``setconc`` -> ``set_species_count`` (exact count); ``addconc`` ->
        ``add_species`` / ``remove_species`` (signed delta). Values resolve via the
        session's own expression evaluator (``session.evaluate`` — the BNG-XML has no
        ``.net`` parameter table), raising if unresolvable (never silently skipped).
      * ``save`` / ``reset`` -> ``save_concentrations`` / ``restore_concentrations``
        (NFsim, in-memory single slot) or ``save_state`` / ``load_state`` (RuleMonkey,
        a binary snapshot file) — the carry-over checkpoint, exactly as BNG2.pl's
        ``saveConcentrations`` / ``resetConcentrations``.

    Method-faithful: a ``nf`` track runs only NFsim segments, an ``rm`` track only
    RuleMonkey. A non-network-free sim step (an ``ode`` / ``ssa`` segment whose end
    state would have to feed the network-free engine — CROSS-ENGINE carry-over, e.g.
    the ``ode -> nf`` / ``ode -> ssa -> nf`` corpus benchmarks) is NOT replayable by a
    session and RAISES, so :func:`run_bngsim_job` falls back to the option-1
    single-segment best-effort path (recording ``replay_error`` — a transparent degrade,
    never worse than baseline).

    Returns ``(result, info)`` where ``result`` is the network-free ``Result`` for the
    representative segment and ``info`` carries ``segments`` and ``rep_method``.
    """
    import bngsim

    if track == "nf":
        session = bngsim.NfsimSession(
            str(xml_path),
            molecule_limit=int(gml) if gml else None,
            block_same_complex_binding=block_same_complex_binding,
        )
    elif track == "rm":
        session = bngsim.RuleMonkeySession(
            str(xml_path),
            molecule_limit=int(gml) if gml else None,
            block_same_complex_binding=block_same_complex_binding,
        )
    else:
        raise RuntimeError(f"multi_segment_replay_netfree got a non-network-free track {track!r}")
    methods = _RM_METHODS if track == "rm" else _NF_METHODS

    snap_dir = Path(tempfile.mkdtemp(prefix=f"replay_{track}_")) if track == "rm" else None
    snap_path = str(snap_dir / "state.bin") if snap_dir else None

    def _value(st):
        """Resolve a setconc/setparam value via the session evaluator, or raise.

        The BNG-XML has no ``.net`` parameter table, so a token like ``"LT"`` (resolved
        to None by parse_protocol's empty net_params) is evaluated against the live
        session namespace. A state mutation must never be silently skipped (GH #177).
        """
        v = st.get("value")
        if v is None:
            tok = st.get("value_token")
            if tok is not None:
                tok = str(tok).strip().strip("'\"")
                if tok:
                    try:
                        v = session.evaluate(tok)
                    except Exception as exc:
                        raise RuntimeError(
                            f"{st['kind']} {st.get('name')!r}: value {tok!r} did not "
                            f"resolve via the session evaluator ({exc})"
                        ) from exc
        if v is None:
            raise RuntimeError(
                f"{st['kind']} {st.get('name')!r}: unresolved value (refusing to skip silently)"
            )
        return float(v)

    captured = None
    n_segments = 0
    try:
        session.initialize(int(seed if seed is not None else 1))
        for i, st in enumerate(steps[: rep_index + 1]):
            kind = st["kind"]
            if kind == "sim":
                if st["method"] not in methods:
                    raise RuntimeError(
                        f"network-free ({track}) replay cannot run a {st['method']!r} "
                        f"segment (cross-engine carry-over is not session-replayable)"
                    )
                r = session.simulate(
                    st["t_start"], st["t_end"], int(st["n_steps"]) + 1, timeout=timeout
                )
                n_segments += 1
                if i == rep_index:
                    captured = r
            elif kind == "setparam":
                session.set_param(st["name"], _value(st))
            elif kind == "setconc":
                session.set_species_count(st["name"], int(round(_value(st))))
            elif kind == "addconc":
                delta = int(round(_value(st)))
                if delta > 0:
                    session.add_species(st["name"], delta)
                elif delta < 0:
                    session.remove_species(st["name"], -delta)
            elif kind == "save":
                session.save_state(snap_path) if track == "rm" else session.save_concentrations()
            elif kind == "reset":
                session.load_state(
                    snap_path
                ) if track == "rm" else session.restore_concentrations()
        if captured is None:
            raise RuntimeError("network-free replay captured no representative segment")
        return captured, {"segments": n_segments, "rep_method": track}
    finally:
        with contextlib.suppress(Exception):
            session.destroy()
        if snap_dir is not None:
            import shutil

            shutil.rmtree(snap_dir, ignore_errors=True)


def _run_bngsim_multiphase_job(
    text: str,
    out_dir: Path,
    bng2_pl: str,
    *,
    track: str,
    timeout: float,
    seed: int | None,
    atol: float,
    rtol: float,
    prefix_info: dict,
) -> dict:
    """Genuine-bngsim run of a ``dirty_carryover`` NETWORK model via full replay (GH #179).

    The option-2 path for a multi-phase network (ode/ssa/psa) model whose representative
    simulate inherits a prior simulate's end state. Generates a CLEAN ``.net`` (declared
    IC — no state prefix; the protocol's setParameter/setConcentration are replayed
    in-process instead), drives bngsim through the full protocol via
    :func:`multi_segment_replay`, and writes the representative segment's
    ``.gdat``/``.cdat`` through the bridge writer (consumer-faithful, same as the
    single-segment direct route). Returns the same provenance shape as
    :func:`run_bngsim_job` with ``protocol_prefix.replayed=True`` and the segment count.
    """
    from bionetgen.core.tools.bngsim_bridge import _write_bngsim_results

    workdir = out_dir / "_netgen"
    artifact, build_sec, err = generate_network(
        text,
        bng2_pl,
        workdir,
        timeout=timeout,
        gen_network=_model_gen_network(text),
        state_prefix="",
    )
    if artifact is None:
        raise RuntimeError(err or "BNG2.pl network generation failed")
    net_params = read_net_parameters(artifact)

    default_method = "ode" if track == "ode" else track
    steps, rep_index = parse_protocol(
        text, methods=_TRACK_METHODS[track], default_method=default_method, net_params=net_params
    )
    if rep_index is None:
        raise RuntimeError(f"could not resolve a {track} representative simulate for replay")
    rep = steps[rep_index]
    if rep["t_end"] is None:
        raise RuntimeError(f"could not resolve a {track} simulate horizon (t_end) for replay")

    result, replay_info = multi_segment_replay(
        artifact,
        steps,
        rep_index,
        track=track,
        atol=atol,
        rtol=rtol,
        seed=seed,
        poplevel=_poplevel(text),
    )
    _write_bngsim_results(result, str(out_dir), "model")

    artifacts = sorted(
        p.name
        for p in out_dir.glob("*")
        if p.is_file() and p.suffix in {".gdat", ".cdat", ".scan"}
    )
    if not artifacts:
        raise RuntimeError("bngsim replay ran but wrote no .gdat/.cdat")
    t_span = (float(rep["t_start"]), float(rep["t_end"]))
    return {
        "engine": "bngsim",
        "method": track,
        "track": track,
        "artifacts": artifacts,
        "t_span": list(t_span),
        "n_points": int(rep["n_steps"]) + 1,
        "build_sec": round(float(build_sec), 3),
        "seed": int(seed) if (track != "ode" and seed is not None) else None,
        # GH #179 provenance: the dirty_carryover model now ran a faithful full-protocol
        # replay (replayed=True), superseding the option-1 best-effort prefix.
        "protocol_prefix": {
            "kept": prefix_info["kept"],
            "dropped_sims": prefix_info["dropped_sims"],
            "dirty_carryover": prefix_info["dirty_carryover"],
            "replayed": True,
            "segments": replay_info["segments"],
            # GH #181: count of seed species re-initialized from a parameter-expression
            # IC during the replay (nonzero only when a setParameter changed a param used
            # in a still-symbolic seed-species IC, e.g. kinetics_mb1n free antigen).
            "reinit_ics": replay_info.get("reinit_ics", 0),
        },
    }


def _run_bngsim_multiphase_netfree_job(
    text: str,
    out_dir: Path,
    bng2_pl: str,
    *,
    track: str,
    timeout: float,
    seed: int | None,
    prefix_info: dict,
) -> dict:
    """Genuine-bngsim run of a ``dirty_carryover`` NETWORK-FREE model via full replay (GH #179).

    The NF/RM analogue of :func:`_run_bngsim_multiphase_job`. Generates a CLEAN BNG-XML
    (no state prefix — the protocol's setParameter/setConcentration are replayed onto the
    live session instead), drives the stateful network-free session through the full
    protocol via :func:`multi_segment_replay_netfree` carrying agent state across segments,
    and writes the representative segment's ``.gdat`` through the bridge writer. Raises (so
    :func:`run_bngsim_job` falls back to the option-1 single-segment path) when the
    protocol is not session-replayable — a cross-engine (ode/ssa -> nf) carry-over or an
    unresolvable species pattern. Returns the same provenance shape with
    ``protocol_prefix.replayed=True``.
    """
    from bionetgen.core.tools.bngsim_bridge import _write_bngsim_results

    workdir = out_dir / "_writexml"
    artifact, build_sec, err = generate_xml(
        text, bng2_pl, workdir, timeout=timeout, state_prefix=""
    )
    if artifact is None:
        raise RuntimeError(err or "BNG2.pl writeXML failed")

    default_method = track
    steps, rep_index = parse_protocol(
        text, methods=_TRACK_METHODS[track], default_method=default_method, net_params={}
    )
    if rep_index is None:
        raise RuntimeError(f"could not resolve a {track} representative simulate for replay")
    rep = steps[rep_index]
    if rep["t_end"] is None:
        raise RuntimeError(f"could not resolve a {track} simulate horizon (t_end) for replay")

    result, replay_info = multi_segment_replay_netfree(
        artifact,
        steps,
        rep_index,
        track=track,
        seed=seed,
        gml=rep.get("gml"),
        timeout=None,
    )
    _write_bngsim_results(result, str(out_dir), "model")

    artifacts = sorted(
        p.name
        for p in out_dir.glob("*")
        if p.is_file() and p.suffix in {".gdat", ".cdat", ".scan"}
    )
    if not artifacts:
        raise RuntimeError("bngsim network-free replay ran but wrote no .gdat")
    t_span = (float(rep["t_start"]), float(rep["t_end"]))
    return {
        "engine": "bngsim",
        "method": track,
        "track": track,
        "artifacts": artifacts,
        "t_span": list(t_span),
        "n_points": int(rep["n_steps"]) + 1,
        "build_sec": round(float(build_sec), 3),
        "seed": int(seed) if seed is not None else None,
        "protocol_prefix": {
            "kept": prefix_info["kept"],
            "dropped_sims": prefix_info["dropped_sims"],
            "dirty_carryover": prefix_info["dirty_carryover"],
            "replayed": True,
            "segments": replay_info["segments"],
            "reinit_ics": 0,  # network-free agent population, no .net seed-species ICs
        },
    }


# --------------------------------------------------------------------------- #
# Matrix (timing+parity) multi-segment helpers (GH #179).
#
# The parity matrix compares bngsim against the LEGACY stack on the SAME model.
# For a dirty_carryover model the single-segment path compares the representative
# run from the .net's baked IC on BOTH engines — they agree with each other but
# neither reflects the model's true multi-phase trajectory. These helpers let the
# matrix drive the FULL protocol: bngsim via multi_segment_replay[_netfree], the
# legacy reference via BNG2.pl run NATIVELY (it handles continue/save/reset).
# --------------------------------------------------------------------------- #
def representative_spec(
    text: str,
    *,
    track: str,
    net_params: dict[str, float],
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
):
    """Continuation-aware representative simulate spec + protocol info for the matrix.

    The matrix replacement for :func:`parse_sim_spec` / :func:`parse_ode_spec` /
    :func:`parse_stoch_spec`: derives the representative from :func:`parse_protocol`
    (continuation-aware ``t_start`` and span — fixing the GH #179 secondary bug where a
    ``continue=>1`` segment is misread as ``[0, t_end]``) and the ``dirty_carryover``
    flag from :func:`state_setup_prefix`. Returns None when no representative simulate
    resolves to a numeric horizon; otherwise ``{t_start, t_end, n_steps, atol, rtol,
    gml, dirty_carryover, steps, rep_index, rep_stmt_idx}``. For a single-phase model
    this matches the old parse_sim_spec selection, so non-dirty rows are unchanged.
    """
    methods = _TRACK_METHODS[track]
    default_method = "ode" if track == "ode" else track
    steps, rep_index = parse_protocol(
        text, methods=methods, default_method=default_method, net_params=net_params
    )
    if rep_index is None:
        return None
    rep = steps[rep_index]
    if rep["t_end"] is None:
        return None
    _, info = state_setup_prefix(text, track=track)
    return {
        "t_start": float(rep["t_start"]),
        "t_end": float(rep["t_end"]),
        "n_steps": int(rep["n_steps"]),
        "atol": float(rep["atol"] or atol),
        "rtol": float(rep["rtol"] or rtol),
        "gml": rep.get("gml"),
        "dirty_carryover": info["dirty_carryover"],
        "steps": steps,
        "rep_index": rep_index,
        "rep_stmt_idx": rep["stmt_idx"],
    }


def _model_body_only(text: str) -> str:
    """The model definition with ALL actions stripped (inverse of :func:`_actions_region`).

    Drops the ``begin actions`` block(s), the top-level actions after ``end model``, and
    any stray top-level action verb — leaving just the model blocks, so a caller can
    append its own (truncated) action protocol. Mirrors :func:`_netgen_bngl`'s stripping
    without appending a ``generate_network`` (the caller supplies the real protocol,
    whose own ``generate_network`` is kept).
    """
    t = re.sub(r"begin\s+actions.*?end\s+actions", "", text, flags=re.DOTALL | re.IGNORECASE)
    m = list(_END_MODEL_RE.finditer(t))
    if m:
        t = t[: m[-1].end()]
    body = "\n".join(ln for ln in t.splitlines() if not _ACTION_LINE_RE.match(ln.strip()))
    return body.rstrip()


# Strip a ``suffix=>"x"`` arg AND its trailing comma (so a leading suffix doesn't leave
# ``{,``); the dangling-comma cleanups below handle a trailing/last-position suffix.
_SUFFIX_ARG_RE = re.compile(r"suffix\s*=>\s*['\"][^'\"]*['\"]\s*,?")
_DANGLING_OPEN_COMMA_RE = re.compile(r"\{\s*,")
_DANGLING_CLOSE_COMMA_RE = re.compile(r",\s*\}")
_SIM_BLOB_CLOSE_RE = re.compile(r"\}\s*\)\s*;?\s*$")


def native_protocol_oracle(
    text: str,
    bng2_pl: str,
    workdir: Path,
    *,
    track: str,
    rep_stmt_idx: int,
    timeout: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Legacy reference for a dirty_carryover model's representative segment (GH #179).

    Runs the model's OWN protocol natively through BNG2.pl (which carries concentrations
    across ``continue`` / honors ``save``/``resetConcentrations``/``setParameter``),
    truncated AFTER the representative simulate (statement ``rep_stmt_idx``) with that
    simulate's ``suffix`` stripped so its output lands on the default ``m`` prefix. The
    representative therefore inherits the prior segments' carry-over exactly as BNG2.pl
    runs the full model. Returns ``(time, values, names)``: the ``.cdat`` SPECIES for the
    deterministic (ode) track, the ``.gdat`` OBSERVABLES for the stochastic (ssa/psa/nf)
    tracks — matching the axis the matrix compares on. ``seed`` (stochastic) is injected
    on every stochastic simulate so each replicate is an independent native trajectory.
    Raises RuntimeError on BNG2.pl failure or missing output.
    """
    statements = _action_statements(_actions_region(text))
    if rep_stmt_idx is None or not (0 <= rep_stmt_idx < len(statements)):
        raise RuntimeError("native oracle: representative statement index out of range")
    kept = list(statements[: rep_stmt_idx + 1])
    stoch = track in (_SSA_METHODS | _PSA_METHODS | _NF_METHODS | {"ssa", "psa", "nf", "rm"})

    def _fix_sim(stmt: str, *, is_rep: bool) -> str:
        if not _SIM_BLOCK_RE.match(stmt):
            return stmt
        if is_rep:
            stmt = _SUFFIX_ARG_RE.sub("", stmt)  # default prefix -> m.cdat / m.gdat
            stmt = _DANGLING_OPEN_COMMA_RE.sub("{", stmt)  # suffix was first arg -> "{,"
            stmt = _DANGLING_CLOSE_COMMA_RE.sub("}", stmt)  # suffix was last arg -> ",}"
            if track == "ode" and "print_CDAT" not in stmt:
                stmt = _SIM_BLOB_CLOSE_RE.sub(",print_CDAT=>1})", stmt)
        if stoch and seed is not None and "seed=>" not in stmt:
            stmt = _SIM_BLOB_CLOSE_RE.sub(f",seed=>{int(seed)}}})", stmt)
        return stmt

    kept = [_fix_sim(s, is_rep=(i == len(kept) - 1)) for i, s in enumerate(kept)]
    bngl = _model_body_only(text) + "\nbegin actions\n" + "\n".join(kept) + "\nend actions\n"

    workdir.mkdir(parents=True, exist_ok=True)
    bngl_path = workdir / "m.bngl"
    bngl_path.write_text(bngl)
    proc = subprocess.run(
        ["perl", bng2_pl, str(bngl_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(workdir),
    )
    if proc.returncode != 0:
        tail = (proc.stderr.strip() or proc.stdout.strip() or "").splitlines()
        raise RuntimeError(f"native protocol oracle failed: {(tail[-1] if tail else '')[:300]}")
    ext = ".cdat" if track == "ode" else ".gdat"
    out = workdir / f"m{ext}"
    if not out.exists():
        # The representative simulate's suffix was stripped above so it writes the
        # default ``m`` prefix. If that file is absent the representative did not
        # actually run — fail loud (the caller records REFERENCE_FAILED) rather than
        # silently substituting some other segment's output (e.g. an earlier
        # ``m_equil.cdat``), which compares mismatched segments and manufactures a
        # false DIFF. GH #69 / #179.
        produced = sorted(p.name for p in workdir.glob(f"*{ext}"))
        raise RuntimeError(
            f"native protocol oracle produced no m{ext} for the representative segment"
            + (f" (the run wrote only {produced})" if produced else " (no output at all)")
        )
    return read_dat(out)


def protocol_session_replayable(steps: list[dict], track: str) -> bool:
    """Whether ``steps`` can be driven by ONE engine's session/Simulator (GH #179).

    A network track (ode/ssa/psa) replay drives a ``.net`` Simulator, which handles any
    network method (ode/ssa/psa) but NOT a network-free segment; a network-free track
    (nf/rm) replay drives a session that runs only its own engine. So a protocol is
    session-replayable iff EVERY simulate segment is on the same side of the
    network/network-free divide as ``track`` — a cross-engine carry-over (e.g.
    ``ode``/``ssa`` -> ``nf``) is not, and the matrix falls back to single-segment.
    """
    want_netfree = track in _NETFREE_TRACKS
    for st in steps:
        if st.get("kind") != "sim":
            continue
        seg_netfree = st["method"] in _NF_METHODS or st["method"] in _RM_METHODS
        if seg_netfree != want_netfree:
            return False
    return True


def _nearest_time_rows(src_time: np.ndarray, src_vals: np.ndarray, target_time) -> np.ndarray:
    """``src_vals`` rows mapped onto ``target_time`` by nearest source time (GH #180).

    The native protocol oracle emits the FULL concatenated trajectory for a ``continue``
    model; this picks the row nearest each representative grid time so the comparison is
    over the representative segment, robust to segment boundaries and non-uniform grids.
    """
    idx = [int(np.argmin(np.abs(src_time - t))) for t in target_time]
    return src_vals[idx]


def bn_stoch_multiseg_ensemble(
    artifact: str | Path,
    sspec: dict,
    *,
    track: str,
    n_rep: int,
    seed_base: int,
    rep_timeout: float | None = None,
    block_same_complex_binding: bool = True,
):
    """bngsim multi-segment ENSEMBLE for a dirty_carryover stochastic model (GH #179).

    Replays the FULL protocol (carry-over) once per seed (``seed_base … seed_base+n_rep-1``)
    and stacks the representative segment's OBSERVABLES. SSA/PSA replay the ``.net`` via
    :func:`multi_segment_replay`; NF/RM replay the BNG-XML via
    :func:`multi_segment_replay_netfree`. Returns
    ``(time, obs[n_rep, n_time, n_obs], names)``. Raises (caller falls back to the
    single-segment ensemble) on a segment the replay can't drive — a cross-engine
    ``ode``/``ssa`` -> ``nf`` carry-over or an unresolvable species pattern.
    """
    times: np.ndarray | None = None
    names: list[str] = []
    out: list[np.ndarray] = []
    for rep in range(n_rep):
        seed = seed_base + rep
        if track in _NETFREE_TRACKS:
            r, _ = multi_segment_replay_netfree(
                artifact,
                sspec["steps"],
                sspec["rep_index"],
                track=track,
                seed=seed,
                gml=sspec.get("gml"),
                timeout=rep_timeout,
                block_same_complex_binding=block_same_complex_binding,
            )
        else:
            r, _ = multi_segment_replay(
                artifact,
                sspec["steps"],
                sspec["rep_index"],
                track=track,
                atol=sspec["atol"],
                rtol=sspec["rtol"],
                seed=seed,
                poplevel=float(sspec.get("poplevel") or 0.0),
            )
        if times is None:
            times = np.asarray(r.time)
            names = list(r.observable_names)
        out.append(np.asarray(r.observables))
    return times, np.stack(out, axis=0), names


def native_stoch_ensemble(
    text: str,
    bng2_pl: str,
    workdir: Path,
    *,
    track: str,
    rep_stmt_idx: int,
    n_rep: int,
    seed_base: int,
    timeout: float,
    target_time: np.ndarray,
):
    """Legacy native-BNG2.pl ENSEMBLE for a dirty_carryover stochastic model (GH #179).

    Runs the model's OWN protocol natively once per seed (:func:`native_protocol_oracle`,
    which carries state across continue/save/reset), reads the representative segment's
    ``.gdat`` OBSERVABLES, and maps them onto ``target_time`` (the bngsim representative
    grid) by nearest time. Returns ``(target_time, obs[n_rep, n_time, n_obs], names)``.
    """
    names: list[str] = []
    out: list[np.ndarray] = []
    for rep in range(n_rep):
        nt, nv, nn = native_protocol_oracle(
            text,
            bng2_pl,
            workdir / f"nat_{rep}",
            track=track,
            rep_stmt_idx=rep_stmt_idx,
            timeout=timeout,
            seed=seed_base + rep,
        )
        out.append(_nearest_time_rows(nt, nv, target_time))
        if not names:
            names = nn
    return np.asarray(target_time), np.stack(out, axis=0), names


def run_bngsim_job(
    run_path: str | Path,
    out_dir: str | Path,
    bng2_pl: str,
    *,
    timeout: float,
    stochastic: bool | None = None,
    seed: int | None = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict:
    """Run ONE corpus job on GENUINE bngsim and write its ``.gdat``/``.cdat`` (GH #175).

    The drop-in, engine-honest replacement for ``bionetgen.run(<bngl>,
    simulator='bngsim')`` used by the sweep/golden generator. Steps:

      1. BNG2.pl generates the network (``.net``) — or BNG-XML (``.xml``) for the
         network-free tracks — from ``run_path``'s model blocks ONLY (no
         simulate, so stock BNG2.pl never invokes ``run_network``/NFsim).
      2. The representative simulate spec (method, horizon, n_steps) is parsed
         from ``run_path`` — exactly as the parity matrix does.
      3. bngsim simulates IN-PROCESS via the bridge's direct route
         (``execute_bngsim_direct_job``), writing the consumer-faithful
         ``model.gdat`` (observables) + ``model.cdat`` (species).

    ``stochastic`` selects the track family; when None (the default) it is
    auto-detected from the model's active methods, matching the sweep's regime
    rule (any non-ODE method -> stochastic). ``seed`` pins the stochastic
    trajectory; when None the baked-in ``seed=>K`` the sweep injected is used
    (else 1). Returns a provenance dict ``{engine:'bngsim', method, track,
    artifacts, t_span, n_points, build_sec, seed}``. Raises on any failure — no
    network, unresolved horizon, an unsupported (pla) method, a missing vendored
    engine, or a bngsim error — so the caller logs a crash instead of an
    empty/mislabelled golden entry. It NEVER substitutes a different engine.
    """
    from bionetgen.core.tools.bngsim_bridge import (
        BNGSIM_HAS_NFSIM,
        BNGSIM_HAS_RULEMONKEY,
        FORMAT_BNG_XML,
        FORMAT_NET,
        BngsimDirectJob,
        execute_bngsim_direct_job,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = Path(run_path).read_text(errors="replace")

    if stochastic is None:
        stochastic = _is_stochastic_text(text)
    if seed is None:
        seed = _parse_seed(text)  # the sweep's baked-in seed=>K (None for ode)

    track = classify_bngsim_track(text, stochastic=stochastic)
    if track is None:
        raise RuntimeError(
            "no bngsim-supported simulate action (only unsupported method(s) such "
            "as 'pla'); refusing to run it on the legacy stack and mislabel bngsim"
        )
    # Hard capability guard — NEVER substitute a different engine for the one the
    # model asked for (the whole point of GH #175). NFsim and RuleMonkey are both
    # vendored into bngsim, so a missing one is a broken build, not a fallback.
    if track == "rm" and not BNGSIM_HAS_RULEMONKEY:
        raise RuntimeError(
            "network-free run requires bngsim's RuleMonkey engine, but this build "
            "reports BNGSIM_HAS_RULEMONKEY=False (RuleMonkey should be vendored in — "
            "rebuild bngsim with RuleMonkey support)"
        )
    if track == "nf" and not BNGSIM_HAS_NFSIM:
        raise RuntimeError(
            "network-free run requires bngsim's NFsim engine, but this build reports "
            "BNGSIM_HAS_NFSIM=False (NFsim should be vendored in — rebuild bngsim "
            "with NFsim support)"
        )

    # GH #177: replay the state-setup actions (setParameter/setConcentration/…) that
    # precede this track's representative simulate so BNG2.pl bakes that state into
    # the .net/.xml the engine consumes (an empty prefix leaves the artifact, and so
    # the result, byte-identical to the strip-everything default).
    state_prefix, prefix_info = state_setup_prefix(text, track=track)

    # GH #179: a multi-phase model whose representative simulate inherits a prior
    # simulate's end state (dirty_carryover) cannot be reproduced by option-1's prefix
    # bake-in — drive the FULL protocol in-process with state carry-over instead. NETWORK
    # tracks (ode/ssa/psa) replay through the .net Simulator; NETWORK-FREE tracks (nf/rm)
    # replay through the stateful NfsimSession / RuleMonkeySession (agent-state carry-over).
    # Surgical gate: single-phase / clean-reset / unaffected models keep the byte-identical
    # single-segment direct path below. A replay that cannot run the protocol (a BNGL
    # species PATTERN bngsim's exact-name set_concentration can't resolve, or a cross-engine
    # ode/ssa -> nf carry-over no session can drive) FALLS BACK to the option-1 best-effort
    # path — never worse than the pre-#179 baseline — recording ``replay_error`` so the
    # audit sees the degrade transparently (not a silent substitution).
    replay_error = ""
    if prefix_info["dirty_carryover"]:
        try:
            if track in _NETFREE_TRACKS:
                return _run_bngsim_multiphase_netfree_job(
                    text,
                    out_dir,
                    bng2_pl,
                    track=track,
                    timeout=timeout,
                    seed=seed,
                    prefix_info=prefix_info,
                )
            return _run_bngsim_multiphase_job(
                text,
                out_dir,
                bng2_pl,
                track=track,
                timeout=timeout,
                seed=seed,
                atol=atol,
                rtol=rtol,
                prefix_info=prefix_info,
            )
        except Exception as exc:
            replay_error = f"{type(exc).__name__}: {exc}"[:300]
            for stale in out_dir.glob("model.*"):  # clear any partial replay artifacts
                if stale.suffix in {".gdat", ".cdat", ".scan"}:
                    stale.unlink(missing_ok=True)

    workdir = out_dir / "_netgen"
    if track in _NETFREE_TRACKS:
        artifact, build_sec, err = generate_xml(
            text, bng2_pl, workdir, timeout=timeout, state_prefix=state_prefix
        )
        fmt = FORMAT_BNG_XML
        net_params: dict[str, float] = {}  # NF/RM read XML; tokens resolve numerically
    else:
        artifact, build_sec, err = generate_network(
            text,
            bng2_pl,
            workdir,
            timeout=timeout,
            gen_network=_model_gen_network(text),
            state_prefix=state_prefix,
        )
        fmt = FORMAT_NET
        net_params = read_net_parameters(artifact) if artifact else {}
    if artifact is None:
        raise RuntimeError(err or "BNG2.pl network generation failed")

    default_method = "ode" if track == "ode" else track
    spec = parse_sim_spec(
        text,
        net_params,
        methods=_TRACK_METHODS[track],
        atol=atol,
        rtol=rtol,
        default_method=default_method,
    )
    if spec is None:
        raise RuntimeError(f"could not resolve a {track} simulate horizon (t_end) from the model")

    method = track  # ode/ssa/psa/nf/rm map 1:1 to the bridge's direct methods
    t_span = (float(spec["t_start"]), float(spec["t_end"]))
    n_points = int(spec["n_steps"]) + 1  # n_steps+1 grid points incl. t_start

    options: dict = {}
    if track == "ode":
        options["rtol"] = float(spec["rtol"])
        options["atol"] = float(spec["atol"])
    else:  # every stochastic track is seeded for a reproducible trajectory
        options["seed"] = int(seed if seed is not None else 1)
        if track == "psa":
            options["poplevel"] = _poplevel(text)
        if track in _NETFREE_TRACKS and spec.get("gml"):
            options["gml"] = int(spec["gml"])

    job = BngsimDirectJob(
        input_path=str(artifact),
        input_format=fmt,
        method=method,
        t_span=t_span,
        n_points=n_points,
        output_dir=str(out_dir),
        output_root="model",
        bngsim_options=options,
    )
    execute_bngsim_direct_job(job)

    artifacts = sorted(
        p.name
        for p in out_dir.glob("*")
        if p.is_file() and p.suffix in {".gdat", ".cdat", ".scan"}
    )
    if not artifacts:
        raise RuntimeError("bngsim ran but wrote no .gdat/.cdat (no observables/species)")
    result = {
        "engine": "bngsim",
        "method": method,
        "track": track,
        "artifacts": artifacts,
        "t_span": list(t_span),
        "n_points": n_points,
        "build_sec": round(float(build_sec), 3),
        "seed": int(seed) if (track != "ode" and seed is not None) else None,
    }
    # GH #177 provenance: record the replayed pre-simulate state-setup so the audit
    # can see which jobs ran a non-default initial config (and flag any that need
    # full multi-phase protocol replay). Omitted when nothing was replayed, so an
    # unaffected job's provenance is unchanged.
    if prefix_info["kept"] or prefix_info["dropped_sims"]:
        result["protocol_prefix"] = {
            "kept": prefix_info["kept"],
            "dropped_sims": prefix_info["dropped_sims"],
            "dirty_carryover": prefix_info["dirty_carryover"],
        }
        # GH #179: a dirty_carryover model reached this single-segment path only because
        # the faithful full-protocol replay could not run it — record why (replayed False
        # + the error) so the audit distinguishes a graceful option-1 best-effort fallback
        # from a clean single-phase model.
        if replay_error:
            result["protocol_prefix"]["replayed"] = False
            result["protocol_prefix"]["replay_error"] = replay_error
    return result


# --------------------------------------------------------------------------- #
# Legacy stochastic adapters — fresh process per seed (per-call cost, no warm
# reuse), reading the .gdat observables and the engine's own self-reported CPU.
# --------------------------------------------------------------------------- #
def run_network_ssa(
    net_path: str | Path,
    run_network_bin: str,
    *,
    t_start: float,
    t_end: float,
    n_steps: int,
    n_rep: int,
    seed_base: int,
    out_prefix: str,
    timeout: float,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """``n_rep`` legacy ``run_network -p ssa`` replicates on the SAME ``.net``.

    Builds the SSA command BNG2.pl would (``-p ssa -h <seed> [-i t_start] --cdat 0
    --fdat 0 -g <net> <net> <step_size> <n_steps>``) and runs it once per seed —
    each call is a fresh process (inherently cold; no warm-solver reuse), so we
    report the per-call wall distribution and run_network's OWN init/propagation CPU
    split (from its stdout). Reads the ``.gdat`` OBSERVABLES (by name) — the
    stochastic comparison axis. Returns ``(time, obs[n_rep,n_time,n_obs], obs_names,
    timing)``; raises RuntimeError if a call fails or produces no ``.gdat``.
    """
    step_size = (t_end - t_start) / n_steps
    times: np.ndarray | None = None
    names: list[str] = []
    out: list[np.ndarray] = []
    walls: list[float] = []
    self_timing: dict = {}
    for rep in range(n_rep):
        pfx = f"{out_prefix}_{rep}"
        cmd = [
            run_network_bin,
            "-o",
            pfx,
            "-p",
            "ssa",
            "-h",
            str(int(seed_base + rep)),
            "-a",
            repr(atol),
            "-r",
            repr(rtol),
        ]
        if t_start != 0.0:
            cmd += ["-i", repr(t_start)]
        cmd += [
            "--cdat",
            "0",
            "--fdat",
            "0",
            "-g",
            str(net_path),
            str(net_path),
            repr(step_size),
            str(int(n_steps)),
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        walls.append(time.perf_counter() - t0)
        if proc.returncode != 0:
            tail = (proc.stderr.strip() or proc.stdout.strip() or "").splitlines()
            raise RuntimeError(f"run_network ssa failed: {(tail[-1] if tail else '')[:300]}")
        if rep == 0:
            self_timing = _parse_run_network_timing(proc.stdout)
        gdat = Path(f"{pfx}.gdat")
        if not gdat.exists():
            raise RuntimeError("run_network ssa produced no .gdat")
        t_arr, vals, nm = read_dat(gdat)
        if times is None:
            times, names = t_arr, nm
        out.append(vals)

    timing = {
        "io_sec": 0.0,
        **_ensemble_stats(walls),
        # run_network's own per-call init/propagation CPU split (rep 0's stdout).
        **self_timing,
        "n_calls": len(walls),
        "config": {
            "codegen": "C (compiled run_network binary)",
            "method": "Gillespie SSA (exact)",
        },
    }
    return times, np.stack(out, axis=0), names, timing


# A ``Species``-type observable forces NFsim to track full complex identity, which
# it refuses to do unless the ``-cb`` (complex book-keeping) flag is set ("you have
# a Species observable … rerun with -cb"). bngsim enables this automatically; the
# legacy NFsim does not, so we detect it from the model and pass ``-cb`` only when
# needed (it carries a runtime cost, so a Molecules-only model is left without it).
_NF_SPECIES_OBS_RE = re.compile(r"^\s*(?:\d+\s+)?Species\b", re.IGNORECASE)


def nf_needs_complex_bookkeeping(bngl_text: str) -> bool:
    """True if the BNGL declares any ``Species``-type observable (needs NFsim ``-cb``)."""
    m = re.search(
        r"begin\s+observables(.*?)end\s+observables", bngl_text, re.DOTALL | re.IGNORECASE
    )
    if not m:
        return False
    return any(_NF_SPECIES_OBS_RE.match(ln) for ln in _strip_comments(m.group(1)).splitlines())


# NFsim self-reports its CPU cost + throughput on stdout, the network-free analogue
# of run_network's init/propagation split.
_NF_CPU_RE = re.compile(r"Total CPU time:\s*([0-9.eE+\-]+)\s*s")
_NF_RPS_RE = re.compile(r"\(\s*([0-9.eE+\-]+)\s*reactions/sec")


def _parse_nfsim_timing(stdout: str) -> dict:
    """Extract NFsim's self-reported total CPU time + reactions/sec from stdout."""
    out: dict[str, float] = {}
    if m := _NF_CPU_RE.search(stdout):
        out["total_cpu_sec"] = float(m.group(1))
    if m := _NF_RPS_RE.search(stdout):
        out["reactions_per_sec"] = float(m.group(1))
    return out


def nfsim_run(
    xml_path: str | Path,
    nfsim_bin: str,
    *,
    t_end: float,
    n_steps: int,
    n_rep: int,
    seed_base: int,
    out_prefix: str,
    timeout: float,
    gml: int | None = None,
    complex_bookkeeping: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """``n_rep`` legacy ``NFsim`` replicates on the SAME BNG-XML.

    Builds the NF command (``-xml <xml> -sim <t_end> -oSteps <n_steps> -seed <seed>
    [-gml <limit>] [-cb] -o <out>``) and runs it once per seed — a fresh process each
    (per-call cost; no warm reuse). ``complex_bookkeeping`` adds ``-cb`` (required
    for a model with a ``Species``-type observable; see
    :func:`nf_needs_complex_bookkeeping`). NFsim has no t_start offset (it simulates
    ``[0, t_end]``), and
    ``-oSteps n_steps`` emits ``n_steps+1`` output rows, matching bngsim's
    ``n_points = n_steps+1`` grid. Reads the ``.gdat`` OBSERVABLES (network-free →
    observables only) and NFsim's own ``Total CPU time`` / ``reactions/sec``.
    Returns ``(time, obs[n_rep,n_time,n_obs], obs_names, timing)``; raises
    RuntimeError on failure.
    """
    times: np.ndarray | None = None
    names: list[str] = []
    out: list[np.ndarray] = []
    walls: list[float] = []
    self_timing: dict = {}
    for rep in range(n_rep):
        outg = f"{out_prefix}_{rep}.gdat"
        cmd = [
            nfsim_bin,
            "-xml",
            str(xml_path),
            "-sim",
            repr(float(t_end)),
            "-oSteps",
            str(int(n_steps)),
            "-seed",
            str(int(seed_base + rep)),
        ]
        if gml:
            cmd += ["-gml", str(int(gml))]
        if complex_bookkeeping:
            cmd += ["-cb"]
        cmd += ["-o", outg]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        walls.append(time.perf_counter() - t0)
        if proc.returncode != 0:
            tail = (proc.stderr.strip() or proc.stdout.strip() or "").splitlines()
            raise RuntimeError(f"NFsim failed: {(tail[-1] if tail else '')[:300]}")
        if rep == 0:
            self_timing = _parse_nfsim_timing(proc.stdout)
        g = Path(outg)
        if not g.exists():
            raise RuntimeError("NFsim produced no .gdat")
        t_arr, vals, nm = read_dat(g)
        if times is None:
            times, names = t_arr, nm
        out.append(vals)

    timing = {
        "io_sec": 0.0,
        **_ensemble_stats(walls),
        **self_timing,
        "n_calls": len(walls),
        "config": {
            "codegen": "C++ (compiled NFsim binary)",
            "method": "NFsim network-free (rejection)",
        },
    }
    return times, np.stack(out, axis=0), names, timing


# --------------------------------------------------------------------------- #
# Per-process warmup (the third cost type, alongside per-model and per-integration)
# --------------------------------------------------------------------------- #
def measure_warmup(run_network_bin: str | None = None) -> dict:
    """One-time per-process warmup for both engines.

    Called once at worker start, before any model runs, so each cost is charged
    here (per-process) rather than to the first model's build.

    - **BNGsim:** ``import bngsim`` (the ``_bngsim_core`` .so load) + the SymPy
      import and one trivial ``sp.diff`` that warms the analytical-Jacobian
      machinery (so per-model ``last_jacobian_sec`` reflects pure derivation).
    - **run_network:** the binary has no import; its per-process cost is the
      process spawn + dynamic-link of the executable, measured as a no-op
      ``run_network`` invocation (``--help``-style exit). Labelled
      ``run_network_source="spawn-proxy"`` so the matrix never conflates it with an
      in-process library import.
    """
    t0 = time.perf_counter()
    import bngsim  # noqa: F401 — the _bngsim_core .so load is part of the measured cost
    import sympy as sp

    _ = sp.diff(sp.Symbol("x") ** 2, sp.Symbol("x"))
    bn_sec = time.perf_counter() - t0

    rn_sec = 0.0
    rn_source = "none"
    if run_network_bin:
        try:
            t1 = time.perf_counter()
            subprocess.run([run_network_bin], capture_output=True, text=True, timeout=30)
            rn_sec = time.perf_counter() - t1
            rn_source = "spawn-proxy"
        except Exception:
            rn_source = "none"
    return {
        "bngsim_sec": round(bn_sec, 6),
        "run_network_sec": round(rn_sec, 6),
        "run_network_source": rn_source,
    }


def measure_stoch_warmup(
    legacy_bin: str | None = None, *, legacy_label: str = "run_network"
) -> dict:
    """One-time per-process warmup for the stochastic tracks (both engines).

    The SSA/NF taxonomy differs from the ODE one (:func:`measure_warmup`): bngsim's
    stochastic paths derive NO analytical Jacobian and import NO SymPy, so its
    warmup is just the ``_bngsim_core`` extension load (the first ``import bngsim``
    in the worker). The legacy engine (``run_network`` for SSA, ``NFsim`` for NF) has
    no in-process import — its per-process cost is the process spawn + dynamic-link,
    measured as a no-op invocation and labelled ``legacy_source="spawn-proxy"`` so
    the matrix never conflates it with a library import. ``legacy_label`` names the
    legacy binary in the report (``run_network`` / ``NFsim``).

    Must be called **once at worker start, before any engine call**, so the
    extension import is charged here (per-process) rather than to the first model's
    per-model load.
    """
    t0 = time.perf_counter()
    import bngsim  # noqa: F401 — the _bngsim_core .so load is the measured cost

    bn_sec = time.perf_counter() - t0

    leg_sec = 0.0
    leg_source = "none"
    if legacy_bin:
        try:
            t1 = time.perf_counter()
            subprocess.run([legacy_bin], capture_output=True, text=True, timeout=30)
            leg_sec = time.perf_counter() - t1
            leg_source = "spawn-proxy"
        except Exception:
            leg_source = "none"
    return {
        "bngsim_sec": round(bn_sec, 6),
        "legacy_sec": round(leg_sec, 6),
        "legacy_source": leg_source,
        "legacy_label": legacy_label,
    }
