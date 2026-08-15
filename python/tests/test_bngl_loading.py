"""``Model.from_bngl`` — BNGL input via BNG2.pl network generation (GH #162).

Two tiers, on purpose.

The first needs no BioNetGen at all: the parts of BNGL loading that are *ours*
are the text transform (which statements reach BNG2.pl) and the resolution/error
surface (what a machine without BNG2.pl is told). Those are the parts that can
regress silently, so they are asserted unconditionally rather than behind a
skip that most environments take.

The second runs ``BNG2.pl`` for real and skips with the resolver's own trail when
it is absent — the ``"BNG2.pl"`` phrasing ``conftest._DECLARED_SKIPS`` already
covers. It asserts the things only a real generator can settle: that a network
comes back and simulates, that the actions block did **not** run, that the
source's ``max_iter`` still bounds the expansion, and that compartmental BNGL
survives the trip (issue #162's open question 3 — it does; BNG2.pl bakes the
volumes into the rate constants exactly as it does for a hand-generated
``.net``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import bngsim
import pytest
from bngsim import Model
from bngsim._bngl_loader import (
    bngl_to_net,
    generate_network_call,
    generation_source,
)
from bngsim._bngpath import resolve_bng
from bngsim._exceptions import ModelError

_REPO = Path(__file__).resolve().parents[2]

_BNG = resolve_bng()
needs_bng2 = pytest.mark.skipif(not _BNG.ok, reason=f"BNG2.pl unavailable: {_BNG.why_not()}")


# A rule set whose network is unbounded without ``max_iter`` — chain assembly by
# A-A binding. Small enough that a couple of iterations run in well under a
# second, which is what makes the truncation assertion cheap.
POLYMER = """\
begin model
begin parameters
  kp 1.0
  km 0.1
end parameters
begin molecule types
  A(a,b)
end molecule types
begin seed species
  A(a,b) 100
end seed species
begin observables
  Molecules Atot A()
end observables
begin reaction rules
  A(a) + A(b) <-> A(a!1).A(b!1)  kp, km
end reaction rules
end model
generate_network({max_iter=>%d})
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


# ─── Tier 1: what reaches BNG2.pl, and what a machine without it is told ──────


def test_the_experiment_is_dropped_from_both_bngl_layouts():
    """The author's experiment goes; nothing inside the model does.

    The two layouts BNGL files actually use — blocks wrapped in ``begin model``,
    and blocks left at top level — put actions at different nesting depths, and a
    filter that handles only one silently ships the other's ``simulate`` calls to
    BNG2.pl.
    """
    wrapped = generation_source(
        "begin model\n"
        "begin parameters\n  k 1\nend parameters\n"
        "end model\n"
        "begin actions\n"
        '  simulate({method=>"ode",t_end=>1e6})\n'
        "  writeSBML()\n"
        "end actions\n"
    )
    bare = generation_source(
        "begin parameters\n  k 1\nend parameters\n"
        '\nsimulate({method=>"ode",t_end=>1e6})\n'
        "writeSBML()\n"
    )
    for out in (wrapped, bare):
        assert "k 1" in out
        assert "parameters" in out
        assert "simulate" not in out
        assert "writeSBML" not in out
        assert "actions" not in out


def test_a_build_directive_before_the_model_survives():
    """``setOption`` above ``begin model`` configures the *generation* and must stay.

    This is the bug the scope-only filter had. `benchmarks/models/bngl/ode/
    catalysis.bngl` opens with `setOption("NumberPerQuantityUnit",6.0221e23)`;
    dropping it left BNG2.pl generating the same topology with every bimolecular
    rate constant off by that factor — a silently wrong model, which is the
    expensive kind. Only the experiment is the author's; a build directive is
    part of the model's definition.
    """
    out = generation_source(
        'version("2.2.4")\n'
        'setOption("NumberPerQuantityUnit",6.0221e23)\n'
        "begin model\nbegin parameters\n  k 1\nend parameters\nend model\n"
        "generate_network({overwrite=>1})\n"
        'simulate({method=>"ode",t_end=>10})\n'
    )
    assert 'setOption("NumberPerQuantityUnit",6.0221e23)' in out
    assert 'version("2.2.4")' in out
    assert "simulate" not in out
    # ...and our own generate_network is appended by the caller, so the source's
    # must not also be here — BNG2.pl would then generate twice.
    assert "generate_network" not in out


def test_state_changes_after_generate_network_are_experiment_setup():
    """The same verb is kept before generation and dropped after it.

    `setConcentration` ahead of `generate_network` seeds the network being built;
    the identical call afterwards is the first line of the author's protocol.
    Position, not verb, is what separates them — and `protocol=True` is where the
    second one is recovered.
    """
    out = generation_source(
        "begin model\nbegin parameters\n  k 1\nend parameters\nend model\n"
        'setConcentration("A(b)",100)\n'
        "generate_network({overwrite=>1})\n"
        'setConcentration("A(b)",0)\n'
    )
    assert out.count("setConcentration") == 1
    assert "100" in out and '"A(b)",0' not in out


def test_a_continued_action_is_dropped_whole():
    """A backslash-continued or multi-line action must not leave a fragment behind.

    Dropping the first physical line only would hand BNG2.pl a dangling
    ``t_end=>10})`` — a parse error blamed on the model rather than on us.
    Statements are classified whole, in both continuation styles BNGL uses.
    """
    backslash = generation_source(
        "begin model\nbegin parameters\n  k 1\nend parameters\nend model\n"
        'simulate({method=>"ode",\\\n'
        "         t_end=>10,\n"
        "         n_steps=>100})\n"
    )
    open_paren = generation_source(
        "begin model\nbegin parameters\n  k 1\nend parameters\nend model\n"
        'simulate({method=>"ode",\n'
        "         t_end=>10,\n"
        "         n_steps=>100})\n"
    )
    for out in (backslash, open_paren):
        assert "k 1" in out
        for fragment in ("simulate", "t_end", "n_steps"):
            assert fragment not in out


def test_a_functions_block_whose_lines_look_like_calls_is_kept():
    """A ``functions`` block's entries are ``name() = expr`` — call-shaped, and inside
    the model. Scope is what distinguishes them from actions; a verb-matching
    filter with no notion of scope would delete the model's own functions."""
    out = generation_source(
        "begin model\nbegin functions\n  rate() = k*Atot\nend functions\nend model\n"
        "generate_network({overwrite=>1})\n"
    )
    assert "rate() = k*Atot" in out
    assert "generate_network" not in out


@pytest.mark.parametrize(
    "source, expected",
    [
        ("generate_network({max_iter=>3})", "generate_network({max_iter=>3,overwrite=>1})"),
        ("generate_network({})", "generate_network({overwrite=>1})"),
        ("generate_network()", "generate_network({overwrite=>1})"),
        ("", "generate_network({overwrite=>1})"),
        # overwrite is appended, so under Perl's last-one-wins hash semantics ours
        # is the one that takes effect.
        (
            "generate_network({overwrite=>0})",
            "generate_network({overwrite=>0,overwrite=>1})",
        ),
    ],
)
def test_generate_network_call_forwards_the_source_options(source, expected):
    """``max_iter``/``max_agg``/``max_stoich`` are how an unbounded rule set is made
    finite — dropping them regenerates a *different model*, or never terminates."""
    text = f"begin model\nbegin parameters\n k 1\nend parameters\nend model\n{source}\n"
    assert generate_network_call(text) == expected


def test_commented_out_generate_network_is_not_read(tmp_path: Path):
    """``#generate_network({max_iter=>1})`` is a comment, not a directive.

    Real corpus files carry commented-out actions (``benchmarks/models/bngl/ode/
    rec_dim_comp.bngl`` is one), and honoring one would silently truncate the
    network to a bound the author had disabled.
    """
    text = "begin model\nend model\n#generate_network({max_iter=>1})\n"
    assert generate_network_call(text) == "generate_network({overwrite=>1})"


def test_missing_file_is_a_filenotfounderror(tmp_path: Path):
    """Reported before BNG2.pl is looked for, so a typo does not read as
    "you have no BioNetGen"."""
    with pytest.raises(FileNotFoundError):
        Model.from_bngl(tmp_path / "nope.bngl")


def test_absent_bng2_names_the_extra_and_the_trail(tmp_path: Path, monkeypatch):
    """The refusal has to be actionable on a box that has three BioNetGens.

    A bare "needs BNG2.pl" is the failure the resolver was written to end: the
    message names every mechanism consulted *and* the fix, so the reader can tell
    "nothing is installed" from "it is somewhere I did not look".
    """
    monkeypatch.delenv("BNG2_PL", raising=False)
    monkeypatch.delenv("BNGPATH", raising=False)
    monkeypatch.setattr(bngsim._bngpath, "_bundled_bngpath", lambda: (None, "not installed"))
    monkeypatch.setattr(bngsim._bngpath.shutil, "which", lambda n, *a, **k: None)

    with pytest.raises(ModelError) as exc:
        Model.from_bngl(_write(tmp_path, "m.bngl", "begin model\nend model\n"))
    msg = str(exc.value)
    assert "bngsim[bngl]" in msg
    for mechanism in ("$BNG2_PL", "$BNGPATH", "PyBioNetGen bundled"):
        assert mechanism in msg


def test_has_bngl_is_a_runtime_probe_not_an_import_check(monkeypatch):
    """``bngsim.HAS_BNGL`` tracks the environment, unlike its import-check neighbours.

    BNG2.pl arrives from ``$BNGPATH`` as legitimately as from an installed
    PyBioNetGen, and a machine with ``bionetgen`` importable but no ``perl``
    cannot load BNGL at all — so a cached or find_spec-based flag would be wrong
    in both directions. It is also lazy: importing bngsim must never pull in
    bionetgen, a 12.8 MB package that brings libroadrunner with it.
    """
    monkeypatch.setattr(bngsim._bngpath, "_bundled_bngpath", lambda: (None, "not installed"))
    monkeypatch.setattr(bngsim._bngpath.shutil, "which", lambda n, *a, **k: None)
    monkeypatch.delenv("BNG2_PL", raising=False)
    monkeypatch.delenv("BNGPATH", raising=False)
    assert bngsim.HAS_BNGL is False
    assert bngsim.capabilities()["features"]["bngl"] is False
    assert "bngsim[bngl]" in bngsim.capabilities()["missing"]["bngl"]


def test_importing_bngsim_does_not_import_bionetgen():
    """The reason ``HAS_BNGL`` is a lazy attribute rather than a module constant."""
    assert "bionetgen" not in sys.modules or bngsim.HAS_BNGL is not None  # sanity
    out = __import__("subprocess").run(
        [sys.executable, "-c", "import bngsim, sys; print('bionetgen' in sys.modules)"],
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "False", out.stderr


def test_capabilities_reports_bngl():
    """``capabilities()["features"]["bngl"]`` is the documented probe, and agrees
    with ``HAS_BNGL``."""
    caps = bngsim.capabilities()
    assert caps["features"]["bngl"] == bngsim.HAS_BNGL
    assert ("bngl" in caps["missing"]) is (not caps["features"]["bngl"])


# ─── Tier 2: BNG2.pl actually runs ───────────────────────────────────────────


@needs_bng2
def test_loads_a_corpus_model_and_simulates(tmp_path: Path):
    """The acceptance case: a rule-based file becomes a simulable model."""
    src = _REPO / "benchmarks" / "models" / "bngl" / "models2" / "egfr_net_red.bngl"
    if not src.is_file():
        pytest.skip("benchmark corpus not present")
    model = Model.from_bngl(src, net_out=tmp_path / "out.net")
    assert model.n_species > 1 and model.n_reactions > 1
    result = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 1.0), n_points=11)
    assert result.species.shape[0] == 11
    assert (tmp_path / "out.net").is_file()


@needs_bng2
def test_load_routes_bngl_to_from_bngl(tmp_path: Path):
    """``Model.load("x.bngl")`` and ``Model.from_bngl`` are the same model."""
    src = _write(tmp_path, "poly.bngl", POLYMER % 2)
    assert Model.load(src).n_species == Model.from_bngl(src).n_species


@needs_bng2
def test_source_max_iter_bounds_the_expansion(tmp_path: Path):
    """A model that says ``max_iter=>1`` means it.

    This rule set has no finite network without a bound, so forwarding is not a
    nicety: regenerating with BNG2.pl's defaults gives a different model at best,
    and at worst does not come back. Two bounds, two sizes, is the assertion that
    the option is read from the source rather than dropped.
    """
    small = Model.from_bngl(_write(tmp_path, "p1.bngl", POLYMER % 1))
    large = Model.from_bngl(_write(tmp_path, "p3.bngl", POLYMER % 3))
    assert 1 < small.n_species < large.n_species


@needs_bng2
def test_actions_block_is_not_executed(tmp_path: Path):
    """The file's own experiment must not run — asserted by an action that would
    fail if it did.

    ``simulate`` before any ``generate_network`` is an error in BNG2.pl ("network
    not initialized"), so a load that succeeds is proof the actions never
    reached it. The cost this protects against is not correctness but time: a
    ``.bngl`` in the wild ends in a ``parameter_scan`` that runs for hours.
    """
    src = _write(
        tmp_path,
        "acts.bngl",
        POLYMER % 2 + 'simulate({method=>"ode",t_end=>1,n_steps=>1})\n',
    )
    assert Model.from_bngl(src).n_species > 1


@needs_bng2
def test_protocol_returns_the_actions_that_were_stripped(tmp_path: Path):
    """``protocol=True`` recovers the experiment rather than discarding it."""
    src = _write(
        tmp_path,
        "proto.bngl",
        POLYMER % 2 + 'simulate({method=>"ode",t_end=>7,n_steps=>10})\n',
    )
    model, spec = Model.from_bngl(src, protocol=True)
    assert model.n_species > 1
    assert [e.t_span for e in spec.experiments] == [(0.0, 7.0)]
    # generate_network is a build directive, not an experiment: dropped cleanly.
    assert "generate_network" in spec.dropped


@needs_bng2
def test_compartmental_bngl_loads(tmp_path: Path):
    """cBNGL survives the round trip (issue #162, open question 3).

    ``rec_dim_comp.bngl`` also exercises two other paths at once: its
    ``generate_network`` is commented out (so the default is used) and its only
    live action is a ``writeMDL()`` that must be stripped.
    """
    src = _REPO / "benchmarks" / "models" / "bngl" / "ode" / "rec_dim_comp.bngl"
    if not src.is_file():
        pytest.skip("benchmark corpus not present")
    model = Model.from_bngl(src)
    assert model.n_species > 1 and model.n_reactions > 1
    assert "Dimers" in model.observable_names


@needs_bng2
def test_initial_conditions_are_as_declared_not_post_experiment(tmp_path: Path):
    """Seed species are what the source's ``seed species`` block declares.

    Running the file as written does *not* give this, and ``toy-jim.bngl`` shows
    both halves of the difference. It declares ``L(r) 0`` (deliberately zero, to
    equilibrate), then equilibrates with ``simulate({...steady_state=>1})`` and
    finally ``setConcentration("L(r)","L_tot")``. BNG2.pl rewrites the ``.net``
    as it goes, so the file that run leaves behind has ``L_tot`` where the model
    said ``0`` and *evaluated numbers* where the model said ``R_tot`` — the
    parameter link severed, taking with it any hope of ``set_param("R_tot", ...)``
    moving the initial condition or of a sensitivity w.r.t. it being right.
    Stripping the experiment is what makes the load reproducible.
    """
    src = _REPO / "benchmarks" / "models" / "bngl" / "models2" / "toy-jim.bngl"
    if not src.is_file():
        pytest.skip("benchmark corpus not present")
    net = bngl_to_net(src, net_out=tmp_path / "toy.net", cache=False)
    species = net.read_text().split("begin species")[1].split("end species")[0]
    seeded = [ln.split()[-1] for ln in species.splitlines() if ln.strip()][:4]
    assert seeded == ["0", "R_tot", "A_tot", "K_tot"], seeded

    model = Model.from_net(net)
    model.set_param("R_tot", 5.0)
    assert model.get_param("R_tot") == 5.0


@needs_bng2
def test_a_pre_model_setoption_reaches_bng2(tmp_path: Path):
    """End-to-end proof for the build-directive rule, on the model that found it.

    `catalysis.bngl`'s `setOption("NumberPerQuantityUnit",6.0221e23)` sets the
    concentration→count conversion, so it multiplies the unit conversion baked
    into every bimolecular rate constant. Dropping it produced a network with the
    right shape and rates ~6e23 too large — which no structural check would
    catch, so the assertion is on the constant itself.
    """
    src = _REPO / "benchmarks" / "models" / "bngl" / "ode" / "catalysis.bngl"
    if not src.is_file():
        pytest.skip("benchmark corpus not present")
    net = bngl_to_net(src, net_out=tmp_path / "catalysis.net", cache=False)
    text = net.read_text()
    assert "unit_conversion=1/(6.0221e+23*volC)" in text
    assert "1.6605503e-12" in text, "the NumberPerQuantityUnit scale is missing"


@needs_bng2
def test_bng2_failure_carries_bng2s_own_words(tmp_path: Path):
    """A broken model reports what BNG2.pl said, not just that something failed.

    Perl diagnostics are the only thing that can localize a BNGL syntax error,
    and BNGsim has no parser of its own to produce a better one.
    """
    src = _write(
        tmp_path,
        "broken.bngl",
        "begin model\nbegin parameters\n  k = = 1\nend parameters\nend model\n",
    )
    with pytest.raises(ModelError) as exc:
        Model.from_bngl(src)
    msg = str(exc.value)
    assert "BNG2.pl" in msg
    assert "broken.bngl" in msg
    assert len(msg) > 120, "the tail of BNG2.pl's output should be attached"


@needs_bng2
def test_timeout_is_reported_as_an_unbounded_network(tmp_path: Path):
    """The runaway case names its own remedy.

    A rule set with no ``max_iter`` can keep BNG2.pl busy indefinitely, and
    "timed out" alone does not tell the reader that the fix is a bound in the
    source rather than a bigger machine.
    """
    unbounded = POLYMER.replace("generate_network({max_iter=>%d})", "generate_network({})")
    src = _write(tmp_path, "runaway.bngl", unbounded)
    with pytest.raises(ModelError, match="timed out"):
        Model.from_bngl(src, timeout=1, cache=False)


# ─── Tier 2: the generated network has to outlive the call ───────────────────


@needs_bng2
def test_generated_net_persists_for_codegen(tmp_path: Path, monkeypatch):
    """``_net_path`` must point at a file that still exists.

    Not housekeeping. ``Model.from_net`` stashes the path, and codegen prefers
    that ``.net`` over the in-memory model precisely because a BNG2.pl network
    carries derived rate-constant parameters (``_rateLaw{N}``) whose chain rules
    the model-based path does not reconstruct (issue #15) — so a scratch
    directory deleted on the way out would leave every from_bngl model with a
    dangling path, failing at simulate time on the models that need the .net
    route most.
    """
    monkeypatch.setattr(bngsim._bngl_loader, "CACHE_DIR", tmp_path / "cache")
    model = Model.from_bngl(_write(tmp_path, "poly.bngl", POLYMER % 2))
    assert model._net_path
    assert Path(model._net_path).is_file()


@needs_bng2
def test_unchanged_bngl_is_served_from_cache(tmp_path: Path, monkeypatch):
    """The second load reuses the network instead of regenerating it."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(bngsim._bngl_loader, "CACHE_DIR", cache)
    src = _write(tmp_path, "poly.bngl", POLYMER % 2)

    first = bngl_to_net(src)
    assert first.parent == cache
    mtime = first.stat().st_mtime_ns

    second = bngl_to_net(src)
    assert second == first
    assert second.stat().st_mtime_ns == mtime, "a cache hit must not rewrite the file"


@needs_bng2
def test_edited_bngl_gets_a_fresh_network(tmp_path: Path, monkeypatch):
    """Content-addressed, so the cache cannot go stale.

    Keying on the flattened model text (not on mtime, and not on the path) is
    what makes reuse safe: an edit that changes the network changes the key, and
    an edit that only touches the actions block correctly hits the same entry.
    """
    cache = tmp_path / "cache"
    monkeypatch.setattr(bngsim._bngl_loader, "CACHE_DIR", cache)
    src = _write(tmp_path, "poly.bngl", POLYMER % 1)
    one = bngl_to_net(src)

    src.write_text(POLYMER % 3)
    three = bngl_to_net(src)
    assert three != one
    assert Model.from_net(one).n_species < Model.from_net(three).n_species

    # ...and an actions-only edit is the same network, so it hits the same entry.
    src.write_text(POLYMER % 3 + 'simulate({method=>"ode",t_end=>1,n_steps=>1})\n')
    assert bngl_to_net(src) == three


@needs_bng2
def test_net_out_and_the_cache_cooperate(tmp_path: Path, monkeypatch):
    """``net_out=`` asks *where* the network goes, not for it to be regenerated.

    Both directions: a run that populates the cache serves a later ``net_out``
    from it, and a ``net_out`` run populates the cache for a later plain load. A
    fitting workflow that wants the ``.net`` as an artifact should not pay for
    generation twice.
    """
    cache = tmp_path / "cache"
    monkeypatch.setattr(bngsim._bngl_loader, "CACHE_DIR", cache)
    src = _write(tmp_path, "poly.bngl", POLYMER % 2)

    first = bngl_to_net(src, net_out=tmp_path / "a.net")
    assert first == tmp_path / "a.net"
    assert len(list(cache.glob("*.net"))) == 1, "a net_out run should still warm the cache"

    cached = next(iter(cache.glob("*.net")))
    stamp = cached.stat().st_mtime_ns
    second = bngl_to_net(src, net_out=tmp_path / "b.net")
    assert second.read_text() == first.read_text()
    assert cached.stat().st_mtime_ns == stamp, "the second run should have been a cache hit"


@needs_bng2
def test_cache_false_still_leaves_a_readable_net(tmp_path: Path, monkeypatch):
    """Opting out of reuse must not reintroduce the dangling-``_net_path`` bug."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(bngsim._bngl_loader, "CACHE_DIR", cache)
    model = Model.from_bngl(_write(tmp_path, "poly.bngl", POLYMER % 2), cache=False)
    assert Path(model._net_path).is_file()
    assert not cache.exists()


@needs_bng2
def test_unwritable_cache_degrades_instead_of_failing(tmp_path: Path, monkeypatch):
    """A read-only cache directory costs reuse, not the load.

    Shared and container filesystems make ``$HOME/.cache`` unwritable often
    enough that failing there would be a support burden with no upside — the
    per-process fallback keeps ``_net_path`` valid, which is the invariant that
    actually matters.
    """
    cache = tmp_path / "cache"
    monkeypatch.setattr(bngsim._bngl_loader, "CACHE_DIR", cache)
    real_publish = bngsim._bngl_loader._publish

    def _read_only_cache(src, dest):
        if dest.parent == cache:
            raise OSError("read-only file system")
        return real_publish(src, dest)

    monkeypatch.setattr(bngsim._bngl_loader, "_publish", _read_only_cache)
    model = Model.from_bngl(_write(tmp_path, "poly.bngl", POLYMER % 2))
    assert Path(model._net_path).is_file()
    assert Path(model._net_path).parent != cache


# ─── Packaging: the extra must not leak into a base install ──────────────────


def _optional_deps() -> dict[str, list[str]]:
    tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+")
    pyproject = _REPO / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml not in this checkout")
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]


def test_bngl_extra_exists_and_names_bionetgen():
    extras = _optional_deps()["optional-dependencies"]
    assert any("bionetgen" in spec for spec in extras["bngl"])


def test_bngl_is_never_a_base_dependency():
    """``pip install bngsim`` must not acquire bionetgen — nor libroadrunner with it.

    bionetgen 0.8.6 requires libroadrunner, seaborn and networkx, so promoting
    this extra would contradict the ``roadrunner`` extra's stated policy that
    RoadRunner is never a base dependency, and would add a Perl runtime
    requirement to a package that runs fine without one.
    """
    project = _optional_deps()
    base = " ".join(project["dependencies"]).lower()
    assert "bionetgen" not in base
    assert "roadrunner" not in base


def test_dev_extra_does_not_pull_the_bngl_extra():
    """``dev`` and the ``parity`` group must not both demand a bionetgen.

    ``[dependency-groups] parity`` pins ``bionetgen @ git+...`` for engine-routing
    provenance; ``bngl`` wants the PyPI release for BNG2.pl. They are different
    requirements for different reasons, and a dev environment asking for both
    hands the resolver a range and a git pin to reconcile. A developer gets
    BNG2.pl from ``--group parity`` or ``$BNGPATH``.
    """
    extras = _optional_deps()["optional-dependencies"]
    assert not any("bngl" in spec for spec in extras["dev"])
