#!/usr/bin/env python3
"""Materialize the bng_parity BNGL corpus from pinned GitHub commits.

Every model resolves to a file in one of three *public* repos at a pinned
commit — no local-checkout, and no PyBioNetGen-report, dependency:

    rulehub      RuleWorld/RuleHub        (also serves the `curated` tutorials)
    rulemonkey   richardposner/RuleMonkey
    bngl_models  wshlavacek/BNGL-Models   (also serves `bngl_library` and the
                                           `curated` Creamer fixture, by subpath)

The corpus *membership* (which 895 models) is read from the committed
``manifest.json`` — its ``(tier, source, relpath)`` triples are stable model
identifiers. This script:

  1. fetches each needed repo at its pin into a gitignored cache (a shallow
     fetch of the exact SHA, ``git fetch --depth 1 origin <sha>``),
  2. resolves every membership entry to its file in the cache,
  3. copies it (plus the companions a *simulation* needs) into
     ``models/<tier>/<source>/<relpath>``,
  4. rewrites ``manifest.json`` / ``manifest.csv`` with provenance
     (``origin.kind`` / ``repo`` / ``commit`` / ``path``), recomputed ``sha256``,
     ``methods``, ``stochastic``, ``has_free``, ``companions``, and per-record
     ``license``. ``manifest.json`` conforms to
     ``parity_checks/manifest.schema.json`` (``{schema_version, corpus,
     generated, upstream_pin, records}``); ``manifest.csv`` stays a flat table.

A developer who already has a checkout *at the pinned commit* can point
``$RULEHUB_DIR`` / ``$RULEMONKEY_DIR`` / ``$BNGL_MODELS_DIR`` at it to skip the
clone; a wrong-commit checkout surfaces as sha256 drift in the summary.

This is I/O-bound (clone + copy + hash) and stays single-threaded on purpose —
nothing here spawns simulators, so it never competes for the box's 6 cores.

Usage:
    python vendor_corpus.py [--membership manifest.json] [--dest models]
                            [--cache DIR] [--no-fetch] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent

# manifest.json conforms to parity_checks/manifest.schema.json.
SCHEMA_VERSION = "1.0"
CORPUS = "bng_parity"
DEFAULT_CACHE = Path(os.environ.get("BNG_PARITY_CACHE", HERE / ".sources_cache"))


@dataclass(frozen=True)
class Pin:
    slug: str  # owner/name, for provenance
    url: str
    sha: str


# The pinned source repos. Bump a SHA here (a deliberate, reviewed change) to
# move the corpus to a newer upstream; re-running then refreshes hashes.
PINS = {
    "rulehub": Pin(
        "RuleWorld/RuleHub",
        "https://github.com/RuleWorld/RuleHub.git",
        "479d6d62a175572f28b2f6b6d7a376b6d1132da2",
    ),
    "rulemonkey": Pin(
        "richardposner/RuleMonkey",
        "https://github.com/richardposner/RuleMonkey.git",
        "0f7011254bddc4843f4426ff7b7ebb315ad8103c",
    ),
    "bngl_models": Pin(
        "wshlavacek/BNGL-Models",
        "https://github.com/wshlavacek/BNGL-Models.git",
        "81c90d8f58354a925651859c94142149afefc4b1",
    ),
    # A SECOND pin of BNGL-Models, at a newer commit, used ONLY to re-source the six
    # house-curated "representative published" models (paper Table S3 / KLU-scaling set)
    # from their corrected, bug-fixed bodies — see CURATED_SIX / resolve(). Kept separate
    # from the `bngl_models` pin so re-sourcing the six does NOT drag the other ~430
    # bngl_models/bngl_library/curated models to a newer upstream (zero collateral): those
    # stay at 81c90d8, these six at dfeb88d.
    "bngl_curated": Pin(
        "wshlavacek/BNGL-Models",
        "https://github.com/wshlavacek/BNGL-Models.git",
        "14ed1b0fafb358a65d327c8da98f1d9743998142",
    ),
}

# SPDX license per source repo (keyed by the resolved pin repo key, so the four
# bench_rulehub/ models — RuleHub-labelled but hosted in BNGL-Models — get the
# BNGL-Models grant). RuleHub/RuleMonkey are MIT; BNGL-Models is CC-BY-4.0 (see
# wshlavacek/BNGL-Models LICENSE). `notice` stays null: the NOTICE-file wiring
# that makes the MIT/CC-BY notice travel with the vendored copy is a separate
# licensing task (currently paused); recording the SPDX here is provenance only.
REPO_LICENSE = {
    "rulehub": {"spdx": "MIT", "notice": None},
    "rulemonkey": {"spdx": "MIT", "notice": None},
    "bngl_models": {"spdx": "CC-BY-4.0", "notice": None},
    "bngl_curated": {"spdx": "CC-BY-4.0", "notice": None},  # same repo as bngl_models
}

# manifest `source` -> (pin repo key, path prefix within that repo). Several
# logical sources live in one repo at different subpaths: bngl_library was
# consolidated under BNGL-Models/bngl_models/, and the Creamer parity fixture
# (a derived model, no upstream simulation protocol) lives in BNGL-Models/creamer/.
SOURCE_TO_REPO = {
    "rulehub": ("rulehub", ""),
    "rulemonkey": ("rulemonkey", ""),
    "bngl_models": ("bngl_models", ""),
    "bngl_library": ("bngl_models", "bngl_models/"),
    "curated": ("bngl_models", "creamer/"),
}


# The six house-curated "representative published" models behind the paper's Supplementary
# Table S3 (cross-engine ODE timing) and the KLU-scaling figure. Their corrected, bug-fixed
# bodies (km21 typo, fyn transphosphorylation fix, added ODE horizons, the Lang v3.2.0 fused
# model, ...) live in wshlavacek/BNGL-Models, NOT upstream in RuleHub as-run — so they are
# RE-SOURCED from BNGL-Models (the `bngl_curated` pin) while keeping their rulehub membership
# IDs stable, exactly like the bench_rulehub fixtures below. Keyed by the membership
# ``(source, relpath)``; ``src`` is the path within the BNGL-Models checkout and ``out``
# (optional) pins the vendored relpath so a pre-existing corpus ID stays byte-stable
# (Barua_2013 keeps its historical ``__PATCHED`` name). Provenance (origin.repo/commit/path
# = BNGL-Models @ dfeb88d / models/…) is recorded per-record; ``test_curated_resource``
# checks these keys stay live.
CURATED_SIX: dict[tuple[str, str], dict] = {
    ("rulehub", "Published/Lang2024/Lang_2024.bngl"): {
        "src": "models/cell_cycle_oscillator_lang2024/cell_cycle_oscillator_lang2024_fused.bngl",
        "reason": "Lang 2024 v3.2.0 full fused RPE-1 cell-cycle model with an in-file ODE horizon",
    },
    ("rulehub", "Published/Kocieniewski2012/Kocieniewski_2012.bngl"): {
        "src": "models/scaffolded_mapk_cascade_kocieniewski2012/scaffolded_mapk_cascade_kocieniewski2012.bngl",
        "reason": "scaffolded MAPK cascade; adds the ODE simulate horizon it lacked upstream",
    },
    ("rulehub", "Published/Barua2007/Barua_2007.bngl"): {
        "src": "models/shp2_regulation_and_function_barua2007/shp2_regulation_and_function_barua2007.bngl",
        "reason": "SHP2 regulation; house-curated body + parameter_scan",
    },
    ("rulehub", "Published/Blinov2006/Blinov_2006.bngl"): {
        "src": "models/combinatorial_egfr_signaling_blinov2006/combinatorial_egfr_signaling_blinov2006.bngl",
        "reason": "combinatorial EGFR; km21 typo corrected 0.01 -> 0.1 /s (Table 1 k-21)",
    },
    ("rulehub", "Published/Barua2013/Barua_2013.bngl"): {
        "src": "models/beta_catenin_destruction_complex_barua2013/beta_catenin_destruction_complex_barua2013.bngl",
        "out": "Published/Barua2013/Barua_2013__PATCHED.bngl",  # keep the historical corpus ID
        "reason": "beta-catenin destruction complex; corrected body (already carries atol, "
        "superseding the former atoll->atol import repair)",
    },
    ("rulehub", "Published/fcerifyn/fceri_fyn.bngl"): {
        "src": "models/early_fceri_signaling_faeder2003/early_fceri_signaling_faeder2003_fyn.bngl",
        "reason": "early FceRI + Fyn; SH2-bound Lyn transphosphorylation bug fixed + ODE horizon added",
    },
}


def resolve(source: str, relpath: str) -> tuple[str, str]:
    """Return (pin repo key, path within that repo) for a membership entry.

    The six CURATED_SIX models are house-curated, bug-fixed re-sources of RuleHub
    ``Published/`` models: they keep their rulehub membership IDs but resolve their
    (corrected) bytes from BNGL-Models at the ``bngl_curated`` pin.

    The four ``rulehub``-sourced ``bench_rulehub/`` models are runtime-tuned
    derived fixtures (equilibration/ligand-add t_end shortened so the NFsim
    parity run is tractable) that don't exist upstream in RuleHub; they're
    hosted in BNGL-Models/bench_rulehub/ with provenance headers. Resolve those
    relpaths there, at the same path, rather than against RuleHub.
    """
    cur = CURATED_SIX.get((source, relpath))
    if cur is not None:
        return "bngl_curated", cur["src"]
    if source == "rulehub" and relpath.startswith("bench_rulehub/"):
        return "bngl_models", relpath
    repo_key, prefix = SOURCE_TO_REPO[source]
    return repo_key, f"{prefix}{relpath}"


# Developer escape hatch: an existing checkout *at the pinned commit*.
ENV_OVERRIDE = {
    "rulehub": "RULEHUB_DIR",
    "rulemonkey": "RULEMONKEY_DIR",
    "bngl_models": "BNGL_MODELS_DIR",
    # Separate from BNGL_MODELS_DIR because the two BNGL-Models pins are at DIFFERENT
    # commits: a single local checkout can only be at one, so testing the curated six
    # against a local checkout uses its own var (point it at a checkout @ the bngl_curated
    # pin; a wrong-commit checkout surfaces as sha256 drift for the six in the summary).
    "bngl_curated": "BNGL_CURATED_DIR",
}

DET_METHODS = {"ode", "cvode"}

# Companion files genuinely needed to *run a simulation* of a model: tfun
# tables, pre-generated networks, species lists, included XML, function/reaction
# includes. Everything else beside a model in its source repo (README.md, CI
# .yaml, .cpp, .png, ...) is that repo's cruft and is NOT vendored. The
# PyBNF/BioNetFit fitting artifacts .conf (job configs) and .exp (fit-target
# data) are deliberately excluded — fitting parity is PyBNF's, not ours.
COMPANION_EXTS = {".tfun", ".net", ".species", ".dat", ".xml", ".func", ".rxn"}

_WIN_ILLEGAL = re.compile(r'[<>:"|?*]')


def windows_safe(relpath: str) -> str:
    """Strip Windows-illegal characters from each component of an output path.

    A no-op for the current corpus (no such names survive in the pinned repos)
    but kept so a future model with a POSIX-only name vendors to a name a
    Windows checkout can hold.
    """
    return str(Path(*[_WIN_ILLEGAL.sub("", p) for p in Path(relpath).parts]))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detect_methods(text: str) -> set[str]:
    t = re.sub(r"#.*", "", text)
    methods: set[str] = set()
    for blob in re.findall(r"simulate\s*\(\s*\{([^}]*)\}", t, re.DOTALL):
        m = re.search(r"method\s*=>\s*['\"]?(\w+)['\"]?", blob)
        methods.add((m.group(1) if m else "ode").lower())
    for meth, _ in re.findall(r"simulate_(\w+)\s*\(\s*\{([^}]*)\}", t, re.DOTALL):
        methods.add(meth.lower())
    for blob in re.findall(r"(?:parameter_scan|bifurcate)\s*\(\s*\{([^}]*)\}", t, re.DOTALL):
        m = re.search(r"method\s*=>\s*['\"]?(\w+)['\"]?", blob)
        methods.add((m.group(1) if m else "ode").lower())
    # "protocol" is a parameter_scan DRIVER (run the named begin/end protocol block
    # for each value), not an integration method: the real methods are the protocol
    # block's own simulate() calls, already captured above. Dropping it keeps a
    # protocol-scan model classified by its actual regime (e.g. nfkb_illustrating_
    # protocols -> ode) instead of being treated as a non-deterministic method.
    methods.discard("protocol")
    return methods


# Known-malformed UPSTREAM models repaired at vendor time. A handful of corpus models
# carry a plain modeler typo upstream (the model is broken, not merely awkwardly named);
# patching the local checkout would only fix this machine, so the repair lives HERE, at
# import, and travels with the vendored copy + its manifest provenance. This is for
# genuine malformations ONLY — a well-formed model with an unfortunate identifier (e.g. a
# parameter named ``lambda``) is NOT repaired; the engine/parity code is taught to cope
# with the name instead. Keyed by ``(source, relpath)``; each edit is COUNT-GUARDED so an
# upstream change (a fix, or a shifted model) fails loudly here rather than silently
# mis-patching or no-op'ing.
CORPUS_REPAIRS: dict[tuple[str, str], dict] = {
    ("bngl_models", "my_models/ode/HarmonicOscillator_redo_v1.bngl"): {
        "issue": "internal#182",
        "reason": (
            "6 setConcentration() species patterns drop the closing paren "
            '(modeler typo): IGF1(ds,hs,label~cold" -> IGF1(ds,hs,label~cold)"'
        ),
        "edits": [
            (
                'setConcentration("IGF1(ds,hs,label~cold"',
                'setConcentration("IGF1(ds,hs,label~cold)"',
                6,
            ),
        ],
    },
    # `atoll`/`abstol`/`reltol` are typo'd tolerance args. BNG2.pl silently IGNORES the
    # bad arg and runs at its DEFAULT tolerance, so the modeler's intended tolerance is
    # silently bypassed — a real correctness defect, not a benign typo. Patch to the
    # canonical atol/rtol so the intended setting is honored (formerly the dead-code
    # SYMBOL_RENAME workaround, now retired into this single mechanism). GH #173/#182.
    ("rulehub", "Published/Barua2009/Barua_2009.bngl"): {
        "issue": "internal#173",
        "reason": "`atoll=>` is a typo for `atol=>` (BNG2.pl ignores it -> default tol)",
        "edits": [("atoll=>", "atol=>", 1)],
    },
    # NB: Barua2013 formerly had an `atoll=>atol` repair here; it is now RE-SOURCED from the
    # house-curated BNGL-Models body (which already uses `atol`) via CURATED_SIX, so the
    # import repair is retired. Its vendored ID keeps the historical `__PATCHED` name via
    # CURATED_SIX[...]["out"].
    ("rulehub", "Examples/physics/phwaveequation/ph_wave_equation.bngl"): {
        "issue": "internal#173",
        "reason": "non-standard tolerance arg names `abstol`/`reltol` (BNG2.pl ignores "
        "them -> default tol); canonical atol/rtol so the intended setting is honored",
        "edits": [("abstol=>", "atol=>", 1), ("reltol=>", "rtol=>", 1)],
    },
    # This derived NFsim-parity fixture (its `parameter_scan` was replaced upstream with an
    # explicit `simulate`) shipped with a stray `par_scan_vals=>[...]` arg that the generator
    # left on the `simulate`. `par_scan_vals` belongs to `parameter_scan`, not `simulate`:
    # BNG2.pl silently ignored it (ran a single NFsim simulate at the n_steps grid), but the
    # PyBioNetGen `bngmodel` parser REJECTS it, silently routing the job to the legacy BNG2.pl
    # stack instead of bngsim. Dropping the arg matches the transform's own single-simulate
    # intent; behavior is identical to what BNG2.pl already ran, only now bngmodel accepts it.
    # Keyed by the MANIFEST source (`rulehub`) even though the pristine bytes resolve from
    # BNGL-Models/bench_rulehub/ (see `resolve()`): apply_corpus_repairs is called with the
    # manifest (source, relpath), and test_corpus_repairs_keys_are_live checks that same key.
    ("rulehub", "bench_rulehub/example3_fit.bngl"): {
        "issue": "internal#173",
        "reason": "stray `par_scan_vals=>[...]` left on the explicit `simulate` (belongs to "
        "`parameter_scan`); BNG2.pl ignored it but PyBioNetGen's bngmodel parser rejects it, "
        "silently routing the job off bngsim to legacy BNG2.pl — dropped to match the "
        "single-simulate intent (behavior identical to what BNG2.pl already ran)",
        "edits": [
            (
                "par_scan_vals=>[0.0005006902,0.001362623,0.0044341334,0.0149210839,\\\n"
                "                0.0441574,0.1507897315,0.5013619944,1.5652727704,5.2257161826,\\\n"
                "                16.9016532291,67.9604112309,213.4593409505],\\\n"
                "                ",
                "",
                1,
            ),
        ],
    },
}


def apply_corpus_repairs(source: str, relpath: str, text: str) -> tuple[str, dict | None]:
    """Repair a known-malformed upstream model. Returns ``(text, record|None)``.

    ``record`` is None when no repair is registered for ``(source, relpath)``. Each edit
    asserts its exact occurrence count; a mismatch raises (``SystemExit``) so an upstream
    change surfaces here for re-verification instead of silently mis-applying.
    """
    spec = CORPUS_REPAIRS.get((source, relpath))
    if spec is None:
        return text, None
    applied = 0
    for find, replace, expected in spec["edits"]:
        n = text.count(find)
        if n != expected:
            raise SystemExit(
                f"corpus repair for {source}:{relpath} expected {expected}x {find!r} "
                f"but found {n} — upstream changed; re-verify the repair ({spec['issue']})"
            )
        text = text.replace(find, replace)
        applied += n
    return text, {"issue": spec["issue"], "reason": spec["reason"], "edits": applied}


# Action calls whose first quoted argument is a species pattern (or option/param name).
# A genuine BNGL malformation in this corpus has been an unbalanced paren in that pattern
# (the HarmonicOscillator typo: ``setConcentration("IGF1(ds,hs,label~cold", ...)``).
_ACTION_QUOTED_ARG = re.compile(
    r"\b(setConcentration|addConcentration|setParameter|saveConcentrations"
    r'|resetConcentrations|setOption)\s*\(\s*"([^"]*)"'
)


def find_malformed_action_args(text: str) -> list[tuple[int, str, str]]:
    """Return ``(line_no, action, arg)`` for each malformed quoted action argument.

    "Malformed" here = unbalanced parentheses/braces in the quoted pattern — a plain
    modeler typo that makes the species/param unresolvable (BNG2.pl may tolerate it, but
    bngsim's exact-name lookups cannot). Comment lines are ignored. This is the scan that
    surfaced the HarmonicOscillator typo; it backs both the import-time defense and the
    corpus CI guard so a re-pin/re-vendor cannot silently reintroduce such a model.
    """
    bad: list[tuple[int, str, str]] = []
    for ln, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for m in _ACTION_QUOTED_ARG.finditer(line):
            arg = m.group(2)
            if arg.count("(") != arg.count(")") or arg.count("{") != arg.count("}"):
                bad.append((ln, m.group(1), arg))
    return bad


def _git(*args: str) -> None:
    r = subprocess.run(
        ["git", *args], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")


def parse_membership(path: Path) -> list[tuple[str, str, str, str | None]]:
    """Return (tier, source, relpath, prior_sha256) per committed manifest entry.

    Accepts both the schema shape (``{..., "records": [...]}``) and the legacy
    bare-array shape, so a re-vendor bootstraps cleanly from either.
    """
    data = json.loads(path.read_text())
    records = data["records"] if isinstance(data, dict) else data
    return [(r["tier"], r["source"], r["relpath"], r.get("sha256")) for r in records]


def ensure_repo(repo_key: str, cache: Path, *, allow_fetch: bool) -> Path:
    """Return a checkout root for ``repo_key`` at its pin (env override, cache, or fetch)."""
    env = os.environ.get(ENV_OVERRIDE[repo_key])
    if env:
        root = Path(env).expanduser()
        if not root.exists():
            raise SystemExit(f"{ENV_OVERRIDE[repo_key]}={root} does not exist")
        return root
    pin = PINS[repo_key]
    dest = cache / f"{repo_key}@{pin.sha[:12]}"
    if (dest / ".git").exists():
        return dest
    if not allow_fetch:
        raise SystemExit(f"{dest} not cached and --no-fetch given")
    print(f"  fetching {pin.slug} @ {pin.sha[:12]} ...", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", str(dest))
    # tolerate a re-run over a partial dir
    subprocess.run(
        ["git", "-C", str(dest), "remote", "remove", "origin"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _git("-C", str(dest), "remote", "add", "origin", pin.url)
    _git("-C", str(dest), "fetch", "--depth", "1", "-q", "origin", pin.sha)
    _git("-C", str(dest), "checkout", "-q", "--detach", "FETCH_HEAD")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--membership", type=Path, default=HERE / "manifest.json")
    ap.add_argument("--dest", type=Path, default=HERE / "models")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--no-fetch", action="store_true", help="use cache/override only; never clone")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    entries = parse_membership(args.membership)
    repos: dict[str, Path] = {}  # repo_key -> checkout root (cloned lazily, once)

    def root_for(repo_key: str) -> Path:
        if repo_key not in repos:
            repos[repo_key] = ensure_repo(repo_key, args.cache, allow_fetch=not args.no_fetch)
        return repos[repo_key]

    manifest: list[dict] = []
    companion_src_dirs: set[Path] = set()
    n_copied = n_missing = n_companions = n_sha_changed = n_repaired = 0
    missing: list[str] = []
    still_malformed: list[str] = []

    for tier, source, relpath, prior_sha in entries:
        repo_key, repo_path = resolve(source, relpath)
        src = root_for(repo_key) / repo_path
        if not src.is_file():
            n_missing += 1
            missing.append(f"{PINS[repo_key].slug}:{repo_path}")
            continue

        text = src.read_text(errors="replace")
        # Repair a known-malformed upstream model (typo) before anything reads it. The
        # vendored file (and detect_methods/has_free) sees the REPAIRED text; sha256 below
        # stays the UPSTREAM hash (provenance + drift), and the repair is recorded in the
        # manifest so the change is auditable, never silent.
        text, repair = apply_corpus_repairs(source, relpath, text)
        rel_in = windows_safe(relpath)
        if repair is not None:
            n_repaired += 1
            # __PATCHED sentinel in the vendored filename (== model_id): a repaired model
            # is impossible to miss downstream — jobs.json/golden/matrix/benchmark all key
            # on model_id, so the patch is visible everywhere with no per-tool awareness.
            # The upstream relpath/origin keep the REAL name (that is what we fetch).
            # Mirrors the existing __FREE sentinel convention.
            p = Path(rel_in)
            rel_in = str(p.with_name(p.stem + "__PATCHED" + p.suffix))
        # Curated re-source (CURATED_SIX): a house-curated body may pin a specific vendored
        # relpath so a pre-existing corpus ID stays byte-stable (Barua_2013 keeps the
        # historical __PATCHED name it earned from the now-retired atoll->atol import repair).
        curated = CURATED_SIX.get((source, relpath))
        if curated is not None and curated.get("out"):
            rel_in = windows_safe(curated["out"])
        rel_out = Path(tier) / source / rel_in
        dest = args.dest / rel_out
        # Defense in depth: a model that vendors still malformed (no repair covered it)
        # is a new broken model — surface it so a repair gets added, never silently shipped.
        if find_malformed_action_args(text):
            still_malformed.append(str(rel_out))
        methods = detect_methods(text)
        digest = sha256(src)
        if prior_sha is not None and digest != prior_sha:
            n_sha_changed += 1

        companions: list[str] = []
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if repair is not None:
                dest.write_text(text)  # repaired content on disk
            else:
                shutil.copy2(src, dest)  # verbatim upstream

        # This model's own bare <stem>.net (a generate_network output / input).
        # Per-simulate <stem>_<suffix>.net snapshots are regenerated each run and
        # not vendored; another model's net belongs to that model.
        own_net = src.with_suffix(".net")
        if own_net.is_file():
            companions.append(own_net.name)
            if not args.dry_run:
                shutil.copy2(own_net, dest.parent / own_net.name)

        # Dir-shared resources (tfun tables, .species/.xml/...): copy once per
        # source directory.
        if src.parent not in companion_src_dirs:
            companion_src_dirs.add(src.parent)
            for sib in sorted(src.parent.iterdir()):
                if (
                    sib.is_file()
                    and sib.suffix.lower() in COMPANION_EXTS
                    and sib.suffix.lower() != ".net"
                ):
                    companions.append(sib.name)
                    if not args.dry_run:
                        shutil.copy2(sib, dest.parent / sib.name)
        n_copied += 1
        n_companions += len(companions)

        entry = {
            # Stable, unique key within the manifest (== model_id downstream).
            "id": str(rel_out),
            "tier": tier,
            "source": source,
            "relpath": windows_safe(relpath),
            "vendored": str(rel_out),
            "sha256": digest,
            "methods": sorted(methods),
            "stochastic": bool(methods - DET_METHODS) or not methods,
            "has_free": "__FREE" in text,
            "companions": companions,
            # Provenance: resolvable on any machine without a local checkout.
            "origin": {
                "kind": "git",
                "repo": PINS[repo_key].slug,
                "commit": PINS[repo_key].sha,
                "path": repo_path,
            },
            "license": REPO_LICENSE[repo_key],
        }
        entry["patched"] = repair is not None
        if repair is not None:
            # ``sha256`` is the upstream hash; the on-disk file is repaired. Record what
            # was changed so the divergence is explicit and auditable.
            entry["repairs"] = [repair]
        # Curated re-source provenance: the membership ID is rulehub, but the bytes come
        # (verbatim) from the house-curated body in BNGL-Models @ the bngl_curated pin. The
        # ``origin`` above already points there; this flag makes the six greppable/auditable
        # (and here sha256 IS the on-disk hash — the file is used verbatim, not repaired).
        entry["curated"] = curated is not None
        if curated is not None:
            entry["curated_source"] = {"repo": PINS[repo_key].slug, "reason": curated["reason"]}
        manifest.append(entry)

    if not args.dry_run:
        # Wrap the per-model records in the unified schema envelope
        # (parity_checks/manifest.schema.json). manifest.csv stays a flat table.
        doc = {
            "schema_version": SCHEMA_VERSION,
            "corpus": CORPUS,
            "generated": date.today().isoformat(),
            "upstream_pin": {
                "sources": {k: {"repo": p.slug, "commit": p.sha} for k, p in PINS.items()}
            },
            "records": manifest,
        }
        (HERE / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n")
        with open(HERE / "manifest.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "tier",
                    "source",
                    "relpath",
                    "repo",
                    "commit",
                    "sha256",
                    "methods",
                    "stochastic",
                    "has_free",
                ]
            )
            for m in manifest:
                w.writerow(
                    [
                        m["tier"],
                        m["source"],
                        m["relpath"],
                        m["origin"]["repo"],
                        m["origin"]["commit"],
                        m["sha256"],
                        "|".join(m["methods"]),
                        m["stochastic"],
                        m["has_free"],
                    ]
                )

    n_stoch = sum(1 for m in manifest if m["stochastic"])
    print(f"membership      : {len(entries)}")
    print(
        f"copied          : {n_copied}  (deterministic {n_copied - n_stoch}, stochastic {n_stoch})"
    )
    print(f"companions      : {n_companions}")
    print(f"has_free        : {sum(1 for m in manifest if m['has_free'])}")
    print(f"repaired        : {n_repaired}  (malformed upstream models patched at import)")
    print(f"sha changed     : {n_sha_changed}  (vs prior manifest)")
    print(f"missing         : {n_missing}")
    for m in missing[:20]:
        print(f"  MISS {m}")
    if still_malformed:
        print(
            f"still malformed : {len(still_malformed)}  (vendored with no repair — ADD a CORPUS_REPAIRS entry)"
        )
        for m in still_malformed[:20]:
            print(f"  MALFORMED {m}")
    if args.dry_run:
        print("(dry run — nothing written)")
    # A model that vendors still malformed (no repair covered it) is a hard failure:
    # it would ship broken syntax into the corpus. So is a missing membership file.
    return 1 if (n_missing or still_malformed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
