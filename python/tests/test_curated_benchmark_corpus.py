"""GH #423 — the manuscript's BNGL models come from the curated collection, once.

Two benchmark suites time the same BNGL models: ``suites/ssa_table5`` runs all
eight at exact-SSA horizons and ``suites/psa`` runs three of them under the
partial-scaling approximation. They used to vendor *separate* pre-generated
``.net`` copies of those three -- three different sha256s for the same models --
and both sets predated the manuscript's re-pointing of its named models at the
house-curated ``wshlavacek/BNGL-Models`` records, so a benchmark re-run would
have faithfully re-measured superseded networks.

What this pins:

1. ``benchmarks/models/curated_nets.json`` describes the artifacts that are
   actually committed -- sha256 and species/reaction counts included.
2. Both suites resolve to *one* ``.net`` per model, the curated one.
3. ``corpus.json`` is the ssa_table5 SSOT: ``_ssa_config.MODELS`` is derived from
   it rather than re-typed, so a horizon cannot be edited in one and not the
   other (the corpus is not on the timing path, so that drift is invisible).
4. The two portability defects that pinned the suite to one developer's machine
   -- an absolute ``run_network`` path and an absolute ``VENV`` in the docs --
   stay fixed.
5. The psa ``Nc`` sweep is declared once, so the runner, its README and its
   emitter cannot disagree about it again.
6. ``regenerate_curated_nets.strip_actions`` drops a record's own protocol
   *completely* while keeping ``generate_network``'s options.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest


def _source_root() -> Path | None:
    """Locate the ``bngsim/`` source tree (mirrors test_version_consistency)."""
    candidates: list[Path] = []
    env_data = os.environ.get("BNGSIM_TEST_DATA")
    if env_data:
        candidates.extend(Path(env_data).resolve().parents)
    env_root = os.environ.get("BNGSIM_SOURCE_ROOT")
    if env_root:
        candidates.append(Path(env_root).resolve())
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        py = candidate / "pyproject.toml"
        if py.is_file() and 'name = "bngsim"' in py.read_text():
            return candidate
    return None


ROOT = _source_root()
BENCH = ROOT / "benchmarks" if ROOT else None
pytestmark = pytest.mark.skipif(
    BENCH is None or not (BENCH / "models" / "curated_nets.json").is_file(),
    reason="benchmarks tree absent in this env",
)


def _manifest() -> dict:
    return json.loads((BENCH / "models" / "curated_nets.json").read_text())


def _load_module(name: str, path: Path):
    """Import a benchmark module by path, with its own directory importable."""
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(path.parent))


def _ssa_config():
    return _load_module(
        "_ssa_config_under_test", BENCH / "suites" / "ssa_table5" / "_ssa_config.py"
    )


def _psa_run():
    return _load_module("psa_run_under_test", BENCH / "suites" / "psa" / "run.py")


def _corpus() -> dict:
    return json.loads((BENCH / "suites" / "ssa_table5" / "corpus.json").read_text())


def _count_block(text: str, name: str) -> int:
    n, inside = 0, False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(f"begin {name}"):
            inside = True
        elif line.startswith(f"end {name}"):
            inside = False
        elif inside and line and not line.startswith("#"):
            n += 1
    return n


# ── 1. the manifest describes what is committed ───────────────────────────────


def test_manifest_matches_the_committed_artifacts():
    man = _manifest()
    assert len(man["models"]) == 8
    for m in man["models"]:
        bngl = BENCH / "models" / m["bngl"]
        net = BENCH / "models" / m["net"]
        assert bngl.is_file(), f"{m['name']}: {m['bngl']} missing"
        assert net.is_file(), f"{m['name']}: {m['net']} missing"
        assert hashlib.sha256(bngl.read_bytes()).hexdigest() == m["source_sha256"], (
            f"{m['name']}: vendored .bngl has drifted from the manifest; re-run "
            f"benchmarks/models/regenerate_curated_nets.py --sync-sources"
        )
        assert hashlib.sha256(net.read_bytes()).hexdigest() == m["net_sha256"], (
            f"{m['name']}: committed .net has drifted from the manifest; re-run "
            f"benchmarks/models/regenerate_curated_nets.py"
        )
        text = net.read_text()
        assert (_count_block(text, "species"), _count_block(text, "reactions")) == (
            m["species"],
            m["reactions"],
        ), f"{m['name']}: manifest sizes do not describe the committed .net"
        # the record's own generate_network options survived into the artifact
        assert m["source"].startswith("models/"), m["source"]


def test_prion_reaches_its_own_stoichiometry_cap():
    # The whole point of keeping generate_network's options: the record raises
    # max_iter over BNG's 100-iteration default so chains reach max_stoich
    # PrP=>120. A bare generate_network truncates at 104/2809, which is exactly
    # what the superseded vendored copy froze.
    prion = next(m for m in _manifest()["models"] if m["name"] == "prion_aggregation")
    assert (prion["species"], prion["reactions"]) == (121, 3843)
    src = (BENCH / "models" / prion["bngl"]).read_text()
    assert "max_iter=>150" in src.replace(" ", "")
    assert "max_stoich=>{PrP=>120}" in src.replace(" ", "")


# ── 2. one artifact set, shared by both suites ────────────────────────────────


def test_both_suites_resolve_the_same_net_for_the_shared_models():
    cfg, psa = _ssa_config(), _psa_run()
    shared = {"tcr_signaling", "erk_activation", "prion_aggregation"}
    assert {m["name"] for m in psa.MODELS} == shared
    for name in shared:
        ssa_path = (cfg.HERE / cfg.MODELS[name]["file"]).resolve()
        psa_path = (psa.NET_DIR / f"{name}.net").resolve()
        assert ssa_path == psa_path, f"{name}: the two suites time different files"
        assert ssa_path.is_file()


def test_neither_suite_vendors_a_net_of_its_own():
    # ssa_table5 kept its eight under suites/ssa_table5/models/bngl/ and psa kept
    # its three under models/net/psa/. Both are gone: models live in
    # benchmarks/models/, per that tree's own rule, and the curated bucket is the
    # one copy these two suites read.
    # Scoped to the two suites this test is about. Globbing every suite failed on
    # any machine that had run ode_fullnet or ode_engines_s3, whose generated
    # .net files are gitignored build products, so it only ever passed in CI.
    strays = [
        p.relative_to(BENCH)
        for suite in ("ssa_table5", "psa")
        for p in (BENCH / "suites" / suite).rglob("*.net")
        if "results" not in p.parts
    ]
    assert not strays, f"a suite is vendoring its own .net again: {strays}"
    for gone in ("models/net/psa", "models/bngl/psa"):
        assert not (BENCH / gone).exists(), (
            f"{gone} is back; it duplicated three curated artifacts byte-for-byte"
        )


def test_psa_sizes_come_from_the_manifest():
    sizes = {m["name"]: (m["species"], m["reactions"]) for m in _manifest()["models"]}
    for m in _psa_run().MODELS:
        assert (m["species"], m["reactions"]) == sizes[m["name"]]


# ── 3. corpus.json is the ssa_table5 SSOT ─────────────────────────────────────


def test_ssa_config_is_derived_from_the_corpus():
    cfg, corpus = _ssa_config(), _corpus()
    expected = {}
    for e in corpus["bngl"]:
        expected[e["name"]] = (e["file"], float(e["t_end"]), int(e["n_steps"]) + 1)
    for e in corpus["sbml"]:
        expected[e["id"]] = (e["file"], float(e["t_end"]), int(e["n_points"]))
    assert set(cfg.MODELS) == set(expected)
    for key, (file, t_end, n_points) in expected.items():
        m = cfg.MODELS[key]
        assert (m["file"], m["t_end"], m["n_points"]) == (file, t_end, n_points), key
        assert (cfg.HERE / m["file"]).is_file(), f"{key}: artifact {m['file']} missing"


def test_corpus_sizes_agree_with_the_curated_manifest():
    sizes = {m["name"]: (m["species"], m["reactions"]) for m in _manifest()["models"]}
    for e in _corpus()["bngl"]:
        assert (e["species"], e["reactions"]) == sizes[e["name"]], e["name"]
        assert e["curated_record"], e["name"]


def test_every_corpus_horizon_is_numeric():
    # A horizon awaiting confirmation from its source paper says so in
    # "t_end_status"; t_end itself is always a number, so no consumer has to
    # carry a placeholder-string fallback.
    for e in _corpus()["bngl"] + _corpus()["sbml"]:
        assert isinstance(e["t_end"], (int, float)), e.get("name") or e.get("id")


# ── 4. portability ────────────────────────────────────────────────────────────


def test_run_network_is_resolved_from_the_environment(monkeypatch):
    monkeypatch.setenv("RUN_NETWORK", "/opt/bng/bin/run_network")
    assert _ssa_config().RUN_NETWORK_BIN == "/opt/bng/bin/run_network"
    monkeypatch.delenv("RUN_NETWORK")
    monkeypatch.setenv("BNGPATH", "/opt/BioNetGen-2.9.3")
    assert _ssa_config().RUN_NETWORK_BIN == "/opt/BioNetGen-2.9.3/bin/run_network"


def test_no_developer_home_paths_in_the_suites():
    # Both defects GH #423 found: an absolute run_network path in _ssa_config.py
    # and an absolute VENV in TIMING_HARNESS.md, neither of which exists on any
    # other machine.
    home = re.compile(r"/Users/(?!\$)[A-Za-z0-9_.-]+/")
    offenders = []
    for suite in ("ssa_table5", "psa"):
        for path in (BENCH / "suites" / suite).rglob("*"):
            if path.suffix not in {".py", ".md", ".json"} or "results" in path.parts:
                continue
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if home.search(line):
                    offenders.append(f"{path.relative_to(BENCH)}:{i}: {line.strip()[:80]}")
    assert not offenders, "hard-coded home directories:\n" + "\n".join(offenders)


# ── 5. the psa Nc sweep is declared once ──────────────────────────────────────


def test_psa_sweep_is_declared_once():
    psa = _psa_run()
    assert psa.POPLEVELS == [10, 30, 100, 300, 1000]
    for m in psa.MODELS:
        assert m["poplevels"] is psa.POPLEVELS, f"{m['name']} carries its own sweep"


def test_psa_docs_describe_the_sweep_that_runs():
    psa = _psa_run()
    n_rows = len(psa.MODELS) * len(psa.POPLEVELS)
    readme = (BENCH / "suites" / "psa" / "README.md").read_text()
    assert ", ".join(str(n) for n in psa.POPLEVELS) in readme, (
        "psa/README.md documents a different Nc sweep than run.POPLEVELS"
    )
    emit_doc = (BENCH / "suites" / "psa" / "emit.py").read_text()
    assert f"{n_rows} rows" in emit_doc, (
        f"psa/emit.py documents a different row count than {n_rows}"
    )
    assert f"{len(psa.POPLEVELS)} Nc values" in emit_doc


# ── 6. generation keeps generate_network and drops everything else ────────────


def _regen():
    return _load_module(
        "regen_curated_under_test", BENCH / "models" / "regenerate_curated_nets.py"
    )


_MULTILINE_MODEL = """begin model
begin parameters
  k 1
end parameters
end model

begin actions
  generate_network({overwrite=>1,check_iso=>1,max_iter=>150,\\
    max_stoich=>{PrP=>120}})
  saveConcentrations()
  resetConcentrations()
  simulate({method=>"ssa",suffix=>"ssa",t_start=>0,t_end=>10,n_steps=>1000,\\
    seed=>2,print_functions=>1})
#  simulate({method=>"ode",suffix=>"ode",t_start=>0,t_end=>10,n_steps=>1000,\\
#    print_functions=>1})
end actions
"""


def test_strip_actions_drops_whole_multiline_actions():
    body = _regen().strip_actions(_MULTILINE_MODEL)
    # The continuation of a stripped simulate must go with it: leaving
    # `seed=>2,print_functions=>1})` behind is a top-level statement that aborts
    # BNG2.pl -- and the abort comes *after* generate_network has already written
    # a .net, so it does not even fail loudly.
    assert "simulate(" not in body.replace("#  simulate(", "")
    assert "seed=>2" not in body
    assert "saveConcentrations" not in body
    assert "resetConcentrations" not in body


def test_strip_actions_keeps_generate_network_options():
    body = _regen().strip_actions(_MULTILINE_MODEL).replace(" ", "")
    assert "max_iter=>150" in body
    assert "max_stoich=>{PrP=>120}" in body
    assert "check_iso=>1" in body


def test_strip_actions_supplies_generate_network_when_absent():
    body = _regen().strip_actions("begin model\nend model\n")
    assert body.count("generate_network") == 1


# ── 7. the emitter survives a partial result set ──────────────────────────────


def _emitter():
    return _load_module(
        "emit_ssa_table_under_test", BENCH / "suites" / "ssa_table5" / "emit_ssa_table.py"
    )


def test_emitter_renders_a_cell_the_results_file_has_no_record_of():
    # `by.get((model, engine), {})` is `{}` for any cell a --only / --engines run
    # or an interrupted sweep did not produce, and the emitter walks all 14x4
    # cells regardless. Indexing r["model"] there raised KeyError, so the emitter
    # could not render anything but a complete matrix.
    emit = _emitter()
    assert emit.display_status({}) == "missing"
    assert emit.normalize_reason({}) == ""


def test_emitter_still_flags_a_cell_that_ran_but_is_unfaithful():
    emit = _emitter()
    (model, engine), why = next(iter(emit.UNFAITHFUL.items()))
    cell = {"model": model, "engine": engine, "status": "ok"}
    assert emit.display_status(cell) == "N/A"
    assert why in emit.normalize_reason(cell)


# ── 8. a declared relaxation is converged, integral, and actually applied ─────


def _relaxed_models() -> list[dict]:
    return [m for m in _manifest()["models"] if m.get("relax")]


def test_a_relaxation_is_declared_with_its_evidence():
    # `relax` is the one thing generation adds beyond generate_network, so it is
    # also the one place hand-tuning could hide. A declaration has to say why,
    # and the why has to carry the convergence evidence.
    for m in _relaxed_models():
        relax = m["relax"]
        assert relax["method"] == "ode", f"{m['name']}: a relaxation must be deterministic"
        assert relax["t_end"] > 0 and relax["n_steps"] > 0
        assert len(relax.get("why", "")) > 200, (
            f"{m['name']}: a relaxation must record why, including its convergence evidence"
        )


def test_relaxed_seeds_are_whole_molecules():
    # An ODE endpoint is a real number, and the engines do not agree on what a
    # fractional molecule count means -- bngsim and run_network round, RoadRunner's
    # gillespie takes it literally and walks the species negative. Integrality is
    # settled in the artifact so every engine reads the same number.
    for m in _relaxed_models():
        text = (BENCH / "models" / m["net"]).read_text()
        inside = False
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("begin species"):
                inside = True
            elif line.startswith("end species"):
                break
            elif inside and line and not line.startswith("#"):
                seed = line.split()[-1]
                assert float(seed) == int(float(seed)) and "." not in seed, (
                    f"{m['name']}: seed {seed!r} is not a whole molecule count"
                )


def test_gene_bursts_carries_the_relaxed_state():
    # Regression on the row this whole mechanism exists for: seeded at 0/0 it
    # measures the basal regime and 12.5% of replicates fire nothing.
    m = next(m for m in _manifest()["models"] if m["name"] == "gene_bursts")
    assert m["relax"]["t_end"] == 360000
    text = (BENCH / "models" / m["net"]).read_text()
    assert "mRNA() 0" in text and "Protein() 467" in text, (
        "gene_bursts lost its relaxed seed; at Table 5's t_end=3600 it would measure "
        "the basal regime instead of the post-relaxation one"
    )


def test_round_species_seeds_leaves_symbolic_seeds_alone():
    regen = _regen()
    net = (
        "begin species\n"
        "    1 A() 4.669798571188e+02\n"
        "    2 B() X_init\n"
        "    3 C() 0.4\n"
        "end species\n"
    )
    out = regen.round_species_seeds(net)
    assert "A() 467" in out
    assert "B() X_init" in out  # a parameter is not ours to round
    assert "C() 0" in out


def test_strip_actions_appends_a_declared_relaxation():
    regen = _regen()
    body = regen.strip_actions(
        _MULTILINE_MODEL, {"method": "ode", "t_end": 360000, "n_steps": 100}
    )
    assert (
        'simulate({method=>"ode",suffix=>"relax",t_start=>0,t_end=>360000,n_steps=>100})' in body
    )
    assert body.rstrip().endswith("writeNetwork({overwrite=>1})")
    # ... and still only for models that declare one
    assert "relax" not in regen.strip_actions(_MULTILINE_MODEL)


# ── 9. suites/ssa reads the shared curated artifacts too ─────────────────────


def _ssa_run():
    return _load_module("ssa_run_under_test", BENCH / "suites" / "ssa" / "run.py")


def test_ssa_suite_shares_the_curated_artifacts():
    # This suite carried byte-identical duplicates of five curated models, which
    # is how three separate tcr_signaling networks came to drift apart.
    curated = {m["name"]: m for m in _manifest()["models"]}
    shared = {
        "gene_expression",
        "gene_expr_3stage",
        "tcr_signaling",
        "erk_activation",
        "prion_aggregation",
    }
    ssa, psa, cfg = _ssa_run(), _psa_run(), _ssa_config()
    for m in ssa.MODELS:
        rec = curated.get(m["name"])
        assert m["curated"] is bool(rec), m["name"]
        if not rec:
            continue
        assert m["name"] in shared
        path = (m["net_dir"] / f"{m['name']}.net").resolve()
        assert path == (BENCH / "models" / rec["net"]).resolve()
        assert (m["species"], m["reactions"]) == (rec["species"], rec["reactions"])
        # every suite that runs this model runs the same bytes
        assert path == (cfg.HERE / cfg.MODELS[m["name"]]["file"]).resolve()
    for m in psa.MODELS:
        assert (psa.NET_DIR / f"{m['name']}.net").resolve() == (
            BENCH / "models" / curated[m["name"]]["net"]
        ).resolve()
    assert {m["name"] for m in ssa.MODELS if m["curated"]} == shared


def test_no_curated_model_keeps_a_copy_in_another_bucket():
    names = {m["name"] for m in _manifest()["models"]}
    strays = [
        p.relative_to(BENCH)
        for p in BENCH.glob("models/net/*/*.net")
        if p.stem in names and p.parent.name != "curated"
    ]
    # models/net/ode/ is a different corpus with a different role (ODE, not SSA);
    # what must not come back is a second copy in an SSA/PSA bucket.
    strays = [p for p in strays if p.parent.name != "ode"]
    assert not strays, f"a curated model has a second SSA/PSA copy again: {strays}"


# ── 9. a horizon and the model it runs come from the same place ───────────────


def _record_ssa_horizon(name: str) -> float | None:
    """t_end of the record's own active exact-SSA action, or None if it has none.

    Read from the vendored copy of the record, whose sha256 the manifest pins to
    the collection, so this is the record's own declaration rather than a number
    re-typed here.
    """
    text = (BENCH / "models" / "bngl" / "curated" / f"{name}.bngl").read_text()
    _, _, actions = text.partition("begin actions")
    joined, buf = [], ""
    for raw in actions.splitlines():
        if raw.rstrip().endswith("\\"):
            buf += raw.rstrip()[:-1]
            continue
        joined.append(buf + raw)
        buf = ""
    horizons = []
    for line in joined:
        stripped = line.strip()
        if stripped.startswith("#") or "simulate(" not in stripped:
            continue
        packed = stripped.replace(" ", "")
        if 'method=>"ssa"' not in packed or "poplevel" in packed:
            continue  # ODE and partial-scaling actions are not this row's protocol
        found = re.search(r"t_end\s*=>\s*([0-9.eE+-]+)", stripped)
        if found:
            horizons.append(float(found.group(1)))
    assert len(horizons) <= 1, f"{name}: more than one active exact-SSA action"
    return horizons[0] if horizons else None


def test_record_horizon_is_what_the_record_declares():
    # GH #425: B10 spent a revision running a 7/10 model at a horizon chosen for
    # the 6/6 artifact it had replaced, and nothing in the corpus compared the
    # two. Every BNGL row now states the record's own horizon, checked here
    # against the record itself, so the pairing cannot go unexamined again.
    for e in _corpus()["bngl"]:
        assert "record_horizon" in e, f"{e['name']}: no record_horizon declared"
        declared = e["record_horizon"]
        actual = _record_ssa_horizon(e["name"])
        if actual is None:
            assert declared is None, (
                f"{e['name']}: record declares no exact-SSA action, so record_horizon is null"
            )
        else:
            assert declared is not None and float(declared) == actual, (
                f"{e['name']}: record_horizon={declared} but the record declares {actual}"
            )


def _renderings(value: float) -> set[str]:
    """How a horizon is plausibly written in prose: 3600000 or 3.6e6."""
    forms = set()
    if float(value).is_integer():
        forms.add(str(int(value)))
    mantissa, _, exponent = f"{float(value):e}".partition("e")
    forms.add(f"{float(mantissa):g}e{int(exponent)}")
    return forms


def test_a_horizon_that_differs_from_the_records_says_so_with_its_number():
    # Running a horizon the record does not declare is allowed -- the manuscript
    # chose several of them -- but the divergence has to be written down, with
    # the record's own value, because that is the difference between a decision
    # and an inheritance. GH #425 is what an inheritance looks like.
    for e in _corpus()["bngl"]:
        declared = e["record_horizon"]
        if declared is None or float(declared) == float(e["t_end"]):
            continue
        caveats = " ".join(e.get("caveats", []))
        assert any(form in caveats for form in _renderings(declared)), (
            f"{e['name']}: runs t_end={e['t_end']} where the record declares "
            f"{declared}, and no caveat names that value"
        )


def test_samoilov_runs_the_records_own_horizon():
    # The #425 decision itself. 0.0018 was the @SIM annotation of the Antimony
    # file the superseded 6/6 artifact was converted from -- a deterministic ODE
    # encoding kept as a stiffness case -- and at that horizon the model has not
    # left the burn-in from its published initial condition.
    e = next(x for x in _corpus()["bngl"] if x["name"] == "samoilov_futile_cycle")
    assert (e["t_end"], e["n_steps"]) == (10, 2000)
    assert e["record_horizon"] == 10
