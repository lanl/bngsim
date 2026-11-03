"""Load a BNGL model by flattening it to a ``.net`` with BNG2.pl (GH #162).

BNGsim has no BNGL parser and does not want one: BNGL is a *rule* language, and
turning rules into the species-and-reactions network BNGsim simulates is network
generation — BNG2.pl's job, and a re-implementation of it would be a second
BioNetGen rather than a reader. So :meth:`bngsim.Model.from_bngl` shells out:
write the model block to a scratch directory, run ``generate_network``, and load
the emitted ``.net`` through the reader that already exists. Same shape as
``bngsim.convert._bng2``, which validates the *writer* through the same oracle.

Three details that are easy to get wrong, and are the reason this is a module
rather than four lines inside ``_model.py``:

**The experiment is stripped before BNG2.pl sees the file, and only the
experiment.** A ``.bngl`` in the wild ends in ``simulate({...})``,
``parameter_scan({...})``, ``writeSBML()`` — and BNG2.pl executes whatever it is
handed. Running the file as written would mean running the author's whole
numerical experiment (minutes to hours, plus ``.cdat``/``.gdat`` spatter) to
obtain a network that ``generate_network`` alone produces in seconds. But
dropping *every* action is the opposite mistake: a ``setOption`` above the model
block configures the generation itself, and losing one silently rescales the
network's rate constants. :func:`generation_source` draws that line, and
``from_bngl(..., protocol=True)`` hands the experiment back as a
:class:`~bngsim.convert.ProtocolSpec` so it is recovered rather than lost.

**The source's own ``generate_network`` options are kept.** ``max_iter``,
``max_agg`` and ``max_stoich`` are how a rule set with an unbounded network is
made finite; a model that says ``generate_network({max_iter=>3})`` means it, and
regenerating with the defaults yields a *different model* (or no termination).
:func:`generate_network_call` recovers those options and forces only
``overwrite``, which is vacuous in a fresh scratch directory anyway.

**The generated ``.net`` outlives the call.** ``Model.from_net`` stashes the
path in ``_net_path``, and deleting the scratch directory would leave every
``from_bngl`` model's ``_net_path`` dangling. Until #803 that was a hard failure:
codegen, and then ``jacobian="jax"``, re-read the file. Nothing does now -- every
backend reads the in-memory model -- but a provenance path that names a missing
file is still a trap for anyone who follows it. So networks land in a
content-addressed cache beside the codegen one, which makes the answer to
"cache it?" a side effect of getting the lifetime right: reloading unchanged
BNGL skips network generation entirely.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from bngsim._exceptions import ModelError

#: BNG2.pl is a network generator, not a solver, but a combinatorial rule set can
#: still keep it busy indefinitely. Ten minutes is long enough for every model in
#: ``benchmarks/models/bngl/`` and short enough that a runaway is reported rather
#: than waited on.
DEFAULT_TIMEOUT = 600


def _default_cache_dir() -> Path:
    """Resolve the generated-network cache, honoring ``BNGSIM_BNGL_CACHE_DIR``.

    Same shape and the same reasons as ``_codegen.CACHE_DIR``: content-addressed,
    so it is shared across processes and can never go stale, and overridable so a
    cluster job can point it at node-local scratch. Networks are small text
    files; like the codegen cache this one is not pruned.
    """
    env = os.environ.get("BNGSIM_BNGL_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "bngsim" / "networks"


CACHE_DIR = _default_cache_dir()

#: Fallback root for when the cache directory is unwritable and no ``net_out``
#: was given. Per-process and removed at exit — long enough for ``_net_path`` to
#: stay valid for every model this process loads, which is the invariant that
#: matters; the cross-process reuse is what is given up.
_SESSION_DIR: Path | None = None


def _session_dir() -> Path:
    global _SESSION_DIR
    if _SESSION_DIR is None:
        _SESSION_DIR = Path(tempfile.mkdtemp(prefix="bngsim-networks-"))
        atexit.register(shutil.rmtree, _SESSION_DIR, True)
    return _SESSION_DIR


# Verbs BNG2.pl must NOT see, even when they precede ``generate_network``.
#
# Everything else at action scope before the source's own generation call is
# *kept*, and that default is the important half. The first cut here dropped
# every action, which silently changed the model: ``benchmarks/models/bngl/ode/
# catalysis.bngl`` opens with ``setOption("NumberPerQuantityUnit",6.0221e23)``
# above its ``begin model``, and without it BNG2.pl generated the same topology
# with every bimolecular rate constant off by that factor — 1e12 where the
# reference network says 1.66e-12. (``convert._protocol`` classifies that same
# option as result-changing for the same reason.) A build directive is not
# something to strip on the way past; only the author's *experiment* is.
#
# So the drop list is exactly the verbs that would run that experiment, write an
# artifact nobody asked for (``writeNET`` would land on top of the network we are
# about to read), redirect the output away from ``--outdir``, or stop BNG2.pl
# before it reaches our appended call.
_EXPERIMENT_VERBS = frozenset(
    {
        "simulate",
        "simulate_ode",
        "simulate_ssa",
        "simulate_nf",
        "simulate_pla",
        "simulate_psa",
        "simulate_protocol",
        "parameter_scan",
        "bifurcate",
    }
)
_GENERATE_VERBS = frozenset({"generate_network", "generate_hybrid_model"})
_ARTIFACT_VERBS = frozenset(
    {
        "writesbml",
        "writemodel",
        "writexml",
        "writemexfile",
        "writemfile",
        "writematlab",
        "writenet",
        "writemdl",
        "writenetwork",
        "writebngl",
        "savemodel",
        "visualize",
        "setoutputdir",
        "quit",
    }
)
_DROPPED_VERBS = _EXPERIMENT_VERBS | _GENERATE_VERBS | _ARTIFACT_VERBS

_VERB_RE = re.compile(r"^([A-Za-z_]\w*)\s*\(")


def _verb_of(statement: str) -> str | None:
    """The lowercased action verb of ``statement``, or ``None`` if it is not a call."""
    m = _VERB_RE.match(statement.strip())
    return m.group(1).lower() if m else None


def generation_source(text: str) -> str:
    """``text`` reduced to what BNG2.pl needs in order to generate the network.

    The model blocks are passed through untouched. At *action scope* — outside
    ``begin model … end model``, or inside an explicit ``begin actions … end
    actions``, which is how :func:`bngsim.convert._protocol._iter_action_calls`
    defines it — statements are kept only if they precede the source's own
    ``generate_network`` and are not in :data:`_DROPPED_VERBS`. Anything at or
    after that call is the author's experiment and goes. A file with no
    ``generate_network`` before its first run is cut at that run, the state it
    starts from (BNG2.pl generates the network there for a network method): a
    ``setParameter`` between two of its runs is a later phase, not part of the
    model.

    Statements are classified whole, so a backslash-continued or multi-line
    action is kept or dropped in one piece rather than leaving a dangling
    ``t_end=>10})`` for BNG2.pl to blame on the model.
    """
    kept: list[str] = []
    stack: list[str] = []
    pending: list[str] = []  # physical lines of an in-progress action statement
    buf = ""  # ...and its comment-stripped text, for classification
    seen_generate = False

    def flush() -> None:
        nonlocal pending, buf, seen_generate
        verb = _verb_of(buf)
        if verb in _GENERATE_VERBS or verb in _EXPERIMENT_VERBS:
            seen_generate = True
        elif verb is not None and not seen_generate and verb not in _DROPPED_VERBS:
            kept.extend(pending)
        pending, buf = [], ""

    for physical in text.splitlines():
        code = physical.split("#", 1)[0].strip()
        low = code.lower()
        if not pending:  # not mid-statement: block delimiters and model bodies
            if low.startswith("begin "):
                block = low.split(None, 1)[1].strip()
                stack.append(block)
                if block != "actions":
                    kept.append(physical)
                continue
            if low.startswith("end "):
                closed = stack.pop() if stack else low.split(None, 1)[1].strip()
                if closed != "actions":
                    kept.append(physical)
                continue
            if stack and stack[-1] != "actions":
                kept.append(physical)
                continue
            if not code:  # blank or comment-only, at action scope
                continue
        pending.append(physical)
        continued = code.endswith("\\")
        buf += code[:-1] if continued else code
        # Complete when nothing is continued and no parenthesis is still open —
        # BNGL actions wrap across lines with and without the backslash.
        if not continued and buf.count("(") <= buf.count(")"):
            flush()
    if pending:
        flush()
    return "\n".join(kept) + "\n"


# The actions after ``generate_network`` that change the state a run starts
# from, which :func:`generation_source` drops with the rest of the experiment
# (issue #1366). Saved and reset as BNG2.pl does, so the no-op
# ``saveConcentrations(); resetConcentrations()`` a file puts before its only
# ``simulate`` changes nothing. Every experiment verb is a run, ``bifurcate``
# too: ``toggle.bngl``'s only run is one.
_SPECIES_SETS = frozenset({"setconcentration", "addconcentration"})
_PARAMETER_SETS = frozenset({"setparameter"})
_SAVES = {"saveconcentrations": "species", "saveparameters": "params"}
_RESETS = {"resetconcentrations": "species", "resetparameters": "params"}


def state_changes_before_first_run(text: str) -> list[tuple[str, list[str]]]:
    """The actions after the source's ``generate_network`` that set the state
    its first run (``simulate``, ``parameter_scan`` or ``bifurcate``) starts
    from, as ``(verb, arguments)`` in source order, each argument unquoted.

    Empty when that run starts from the model as loaded: no generation call
    before it (:func:`generation_source` cuts at the run), no run after it, no
    change before the
    run, or only changes undone by a
    ``resetConcentrations``/``resetParameters`` to a state saved before them.
    A reset that undoes a change made *before* the generation call, which the
    loaded network holds, is one of the actions returned.
    """
    from bngsim.convert._protocol import _iter_action_calls, _split_top_level

    calls = [
        (verb, [a.strip().strip("\"'") for a in _split_top_level(args)])
        for verb, args in _iter_action_calls(text)
    ]
    # Where generation_source cuts: the first generation call, or a run before one.
    cut = _GENERATE_VERBS | _EXPERIMENT_VERBS
    first = next((i for i, (verb, _) in enumerate(calls) if verb.lower() in cut), None)
    if first is None or calls[first][0].lower() in _EXPERIMENT_VERBS:
        return []
    # Each kind of state: the changes in effect since the model as declared (by
    # index into *calls*), and those saved under each label, as BNG2.pl keeps
    # them. A reset to a label never saved puts back the model as declared: it
    # does for the unlabeled default, and any other is an error that stops the
    # file before its run.
    state: dict[str, list[int]] = {"species": [], "params": []}
    saved: dict[str, dict[str, list[int]]] = {"species": {}, "params": {}}
    loaded: dict[str, list[int]] = {}  # what generation_source applies
    last_reset: dict[str, int] = {}
    for i, (verb, args) in enumerate(calls):
        low = verb.lower()
        label = args[0] if args else "DEFAULT"
        if i == first:
            loaded = {kind: list(changes) for kind, changes in state.items()}
        elif i > first and low in _EXPERIMENT_VERBS:
            out: list[int] = []
            for kind, changes in state.items():
                out += [c for c in changes if c not in loaded[kind]]
                if any(c not in changes for c in loaded[kind]):
                    out.append(last_reset[kind])
            return [calls[c] for c in sorted(out)]
        elif low in _SPECIES_SETS:
            state["species"].append(i)
        elif low in _PARAMETER_SETS:
            state["params"].append(i)
        elif low in _SAVES:
            saved[_SAVES[low]][label] = list(state[_SAVES[low]])
        elif low in _RESETS:
            state[_RESETS[low]] = list(saved[_RESETS[low]].get(label, []))
            last_reset[_RESETS[low]] = i
    return []


def _value(model, text: str) -> float | None:
    """*text* evaluated in the model's arithmetic, with its parameters as loaded."""
    try:
        return float(model._core.evaluate_over_times(text, [0.0])[0])
    except Exception:
        return None


def _network_spelling(pattern: str) -> str:
    """A one-compartment species pattern as the network spells it. BNGL actions
    write ``@c:A()`` or ``A()@c``; BNG2.pl's network, and the names a model loads
    from it, write ``@c::A()``."""
    m = re.fullmatch(r"\s*@(\w+)::?(.*)", pattern, re.S)
    if m is not None:
        return f"@{m.group(1)}::{m.group(2).strip()}"
    # A()@c, or a complex whose molecules all carry the one suffix: A(b!1)@c.B(a!1)@c
    parts = [re.fullmatch(r"\s*([^@]*?)\s*@(\w+)\s*", part, re.S) for part in pattern.split(".")]
    if all(parts) and len({m.group(2) for m in parts if m}) == 1:
        compartment = parts[0].group(2) if parts[0] else ""
        return f"@{compartment}::" + ".".join(m.group(1) for m in parts if m)
    return pattern.strip()


def _molecule_key(pattern: str) -> tuple[str | None, str, tuple[str, ...]] | None:
    """A one-molecule pattern's compartment, name and sorted components: the
    network names a species with its components in canonical order,
    ``LPS(CD14,LPS,MD2,TLR4)`` for an action's ``LPS(MD2,TLR4,CD14,LPS)``. A
    fixed species' ``$`` is not part of it."""
    text = _network_spelling(pattern)
    compartment = None
    m = re.fullmatch(r"@(\w+)::(.*)", text, re.S)
    if m is not None:
        compartment, text = m.group(1), m.group(2)
    m = re.fullmatch(r"\s*\$?\s*([A-Za-z_]\w*)\s*(?:\((.*)\))?\s*", text, re.S)
    if m is None or "." in text:
        return None
    parts = [c.strip() for c in (m.group(2) or "").split(",") if c.strip()]
    return compartment, m.group(1), tuple(sorted(parts))


def _species_name(model, pattern: str) -> str | None:
    """The network's name for the species *pattern* names, or ``None`` if the
    network has no such species: the pattern itself, or spelled as the network
    spells a compartment (``@c::A()`` for ``@c:A()`` or ``A()@c``), either one
    with or without a fixed species' ``$`` as :meth:`~bngsim.Model.get_concentration`
    reads it, or a one-molecule species with its components in another order."""
    for candidate in dict.fromkeys((pattern, _network_spelling(pattern))):
        try:
            return model._species_call(candidate, model._core.get_concentration)[0]
        except ModelError:
            pass
    key = _molecule_key(pattern)
    if key is not None:
        for name in model.species_names:
            if _molecule_key(name) == key:
                return name
    return None


def _target(model, verb: str, args: list[str]) -> tuple[str, str] | None:
    """What a ``set*``/``add*`` action writes in *model*, ``("param", name)`` or
    ``("species", the network's name)``, or ``None`` if that cannot be read."""
    if len(args) != 2:
        return None
    if verb.lower() in _PARAMETER_SETS:
        return ("param", args[0]) if args[0] in model.param_names else None
    name = _species_name(model, args[0])
    return None if name is None else ("species", name)


def _keeps_loaded_value(model, target: tuple[str, str], actions) -> bool:
    """Whether *actions*, the last set of *target* and the adds after it, leave
    it at the value *model* holds."""
    kind, name = target
    held = float(model.get_param(name) if kind == "param" else model.get_concentration(name))
    value = held
    for verb, (_, text) in actions:
        new = _value(model, text)
        if new is None:
            return False
        value = value + new if verb.lower() == "addconcentration" else new
    return value == held


def dropped_state_changes(model, text: str) -> list[str]:
    """The actions of :func:`state_changes_before_first_run` that would start the
    first run somewhere other than *model*, loaded without them, starts: each
    written as in the source, in source order. They are judged by target, as
    the run meets them: a parameter's last ``setParameter``, and a species' last
    ``setConcentration`` with the ``addConcentration`` calls after it, are named
    only if they leave a value other than the loaded one. One whose effect
    cannot be read (a pattern the network does not name, a value that does not
    evaluate, a reset) counts as a change.
    """
    changes = state_changes_before_first_run(text)
    # Each target's actions in effect at the run, by index into *changes*.
    in_effect: dict[tuple[str, str], list[int]] = {}
    named: list[int] = []
    for i, (verb, args) in enumerate(changes):
        target = _target(model, verb, args)
        if target is None:
            named.append(i)
        elif verb.lower() == "addconcentration":
            in_effect.setdefault(target, []).append(i)
        else:
            in_effect[target] = [i]
    for target, indices in in_effect.items():
        if not _keeps_loaded_value(model, target, [changes[i] for i in indices]):
            named += indices
    return [
        f"{verb}({', '.join(repr(a) for a in args)})"
        for verb, args in (changes[i] for i in sorted(named))
    ]


def generate_network_call(text: str) -> str:
    """The ``generate_network`` action to run, honoring the source's own options.

    Returns the source's last ``generate_network(...)`` with ``overwrite=>1``
    appended (appended, not prepended, so it wins a duplicate key under Perl's
    last-one-wins hash semantics). When the file has none — a network-free model
    that only ever calls ``simulate_nf``, say — the bare default is used, and the
    caller's timeout is what bounds an unbounded rule set.
    """
    from bngsim.convert._protocol import _iter_action_calls

    args = ""
    for verb, argstr in _iter_action_calls(text):
        if verb.lower() == "generate_network":
            args = argstr.strip()

    if args.startswith("{") and args.endswith("}"):
        inner = args[1:-1].strip().rstrip(",")
        return "generate_network({" + (f"{inner},overwrite=>1" if inner else "overwrite=>1") + "})"
    # Empty, or the rarely-seen positional form — neither carries options worth
    # forwarding, so fall back to the default rather than guess at a rewrite.
    return "generate_network({overwrite=>1})"


def resolve_bng2(explicit: str | Path | None = None) -> Path:
    """BNG2.pl for a BNGL load, or :class:`~bngsim.ModelError` naming the trail.

    Not an ``ImportError``: BNG2.pl arrives from ``$BNG2_PL``/``$BNGPATH`` just as
    legitimately as from an installed PyBioNetGen, so "the ``bngl`` extra is not
    installed" is one cause among several rather than the diagnosis. The message
    lists every mechanism consulted (and names the extra among the fixes), which
    is the difference between "you have no BioNetGen" and "it is somewhere I did
    not look".
    """
    from bngsim._bngpath import resolve_bng

    r = resolve_bng(explicit)
    if not r.ok:
        raise ModelError(f"cannot load BNGL: {r.why_not()}")
    assert r.bng2_pl is not None  # implied by r.ok; for type narrowing
    return r.bng2_pl


def _cache_key(effective_bngl: str, bng2: Path) -> str:
    """Digest of everything that determines the generated network.

    The flattened BNGL *and* the generator: two BioNetGen releases do not have to
    agree on what a rule set expands to, so an upgrade — in place or to a
    different install — must not be served a network the previous one produced.
    """
    h = hashlib.sha256()
    h.update(effective_bngl.encode())
    h.update(b"\0")
    h.update(str(bng2).encode())
    try:
        st = bng2.stat()
        h.update(f"\0{st.st_size}\0{st.st_mtime_ns}".encode())
    except OSError:  # pragma: no cover - the file was just resolved
        pass
    return h.hexdigest()[:32]


def _publish(src: Path, dest: Path) -> Path:
    """Move ``src`` onto ``dest`` atomically, as the codegen cache does.

    Concurrent loads of the same model race on the same destination; each writes
    a uniquely named neighbour and ``os.replace()``s it into place, so a reader
    can never observe a half-written network. The token covers threads as well as
    processes — a parallel fitting run fans out both ways.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_name(f"{dest.name}.{uuid.uuid4().hex[:12]}.tmp")
    shutil.copyfile(src, staging)
    os.replace(staging, dest)
    return dest


def bngl_to_net(
    path: str | Path,
    *,
    bng2_pl: str | Path | None = None,
    net_out: str | Path | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    cache: bool = True,
) -> Path:
    """Flatten a ``.bngl`` to a ``.net`` with ``BNG2.pl generate_network``.

    Returns the path of the generated network, which persists (see the module
    docstring on ``_net_path``): ``net_out`` when given, else the content-
    addressed entry under :data:`CACHE_DIR`, else — when that is unwritable or
    ``cache=False`` — a per-process directory removed at interpreter exit.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BNGL file not found: {path}")
    bng2 = resolve_bng2(bng2_pl)
    source = path.read_text()
    effective = generation_source(source) + "\n" + generate_network_call(source) + "\n"

    stem = path.stem or "model"
    key = _cache_key(effective, bng2)
    cached = CACHE_DIR / f"{stem}_{key}.net"
    if cache and cached.is_file():
        # A cache hit still serves a net_out= caller: they asked for the network
        # at a path of their choosing, not for it to be regenerated.
        return cached if net_out is None else _publish(cached, Path(net_out))

    workdir = Path(tempfile.mkdtemp(prefix="bngsim-bngl-"))
    try:
        scratch = workdir / f"{stem}.bngl"
        scratch.write_text(effective)
        net = _run_bng2(bng2, scratch, workdir, timeout=timeout, source_name=path.name)

        if net_out is not None:
            # Populate the cache too, so a later plain load reuses this run.
            if cache:
                with contextlib.suppress(OSError):
                    _publish(net, cached)
            return _publish(net, Path(net_out))
        if cache:
            try:
                return _publish(net, cached)
            except OSError:
                # An unwritable cache is a degradation, not a failure: fall back
                # to the per-process directory so the load still succeeds and
                # _net_path still points at a real file.
                pass
        return _publish(net, _session_dir() / f"{stem}_{key}.net")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run_bng2(bng2: Path, scratch: Path, workdir: Path, *, timeout: int, source_name: str) -> Path:
    """Run ``BNG2.pl`` on ``scratch`` and return the ``.net`` it wrote."""
    try:
        proc = subprocess.run(
            ["perl", str(bng2), "--outdir", str(workdir), str(scratch)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workdir),
        )
    except subprocess.TimeoutExpired as e:
        raise ModelError(
            f"BNG2.pl generate_network timed out after {timeout}s on {source_name} — "
            "the rule set may generate an unbounded network. Bound it in the "
            "source (max_iter/max_agg/max_stoich on generate_network, which "
            "from_bngl forwards), raise timeout=, or simulate it network-free "
            "with NFsim instead."
        ) from e

    # `--outdir` puts the network beside the scratch file, but the source's own
    # generate_network may carry prefix=>/suffix=>, which renames it — so look
    # for the expected name and fall back to whatever single .net appeared.
    net = workdir / f"{scratch.stem}.net"
    if not net.is_file():
        found = sorted(workdir.glob("*.net"))
        if len(found) == 1:
            net = found[0]
    if not net.is_file():
        tail = ((proc.stdout or "")[-1200:] + (proc.stderr or "")[-600:]).strip()
        raise ModelError(
            f"BNG2.pl generated no network from {source_name} (exit {proc.returncode}). "
            f"BNG2.pl said:\n{tail}"
        )
    return net


def load_bngl(
    path: str | Path,
    *,
    bng2_pl: str | Path | None = None,
    net_out: str | Path | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    cache: bool = True,
    defer_jacobian: bool | None = None,
):
    """Generate the network for ``path`` and load it through ``Model.from_net``."""
    from bngsim._model import Model

    net = bngl_to_net(path, bng2_pl=bng2_pl, net_out=net_out, timeout=timeout, cache=cache)
    return Model.from_net(net, defer_jacobian=defer_jacobian)
