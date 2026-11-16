"""A perl on PATH that cannot run BNG2.pl is reported as such (issue #1761).

On a Debian slim image ``/usr/bin/perl`` is ``perl-base``, without the core
library that BNG2.pl's line 34 (``use FindBin``) and its ``Perl2`` modules load.
``shutil.which("perl")`` said yes there, so ``HAS_BNGL`` read ``True`` and
``Model.from_bngl`` died inside BNG2.pl with "Can't locate FindBin.pm". The
resolver now asks that perl to compile the script it found (``perl -c``), and
its refusal relays Perl's first line and names the fix.

Oracles: the real perl on this machine compiling a stand-in BNG2.pl that asks
for a module no Perl has, which fails on every host the way perl-base fails on
the image; the image's own first line (lanl/bngsim#1694's Linux pass, run
43671106019), fed through a stubbed subprocess so the text handling needs no
perl; and the resolver's three callers agreeing on the answer.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import bngsim
import pytest
from bngsim import _bngpath as bngpath
from bngsim._exceptions import ModelError

needs_perl = pytest.mark.skipif(
    shutil.which("perl") is None, reason="BNG2.pl / perl not available"
)

# A module no Perl installation has, so the stand-in fails to compile everywhere
# the way BNG2.pl fails on perl-base; the name is what the message must relay.
MISSING = "No::Such::Module::ForBngsim1761"

# What perl-base 5.40.1 printed on python:3.13-slim for BNG2.pl (run 43671106019),
# with its @INC listing abridged.
SLIM_STDERR = (
    "Can't locate FindBin.pm in @INC (you may need to install the FindBin module) "
    "(@INC entries checked: /etc/perl /usr/local/lib/x86_64-linux-gnu/perl/5.40.1 "
    "/usr/local/share/perl/5.40.1 /usr/lib/x86_64-linux-gnu/perl5/5.40 /usr/share/perl5 "
    "/usr/lib/x86_64-linux-gnu/perl-base) at /tmp/v/lib/python3.13/site-packages/"
    "bionetgen/bng-linux/BNG2.pl line 34.\n"
    "BEGIN failed--compilation aborted at /tmp/v/lib/python3.13/site-packages/"
    "bionetgen/bng-linux/BNG2.pl line 34.\n"
)


@pytest.fixture(autouse=True)
def _only_bngpath(monkeypatch):
    """Resolve through ``$BNGPATH`` alone, with the probe's cache empty."""
    for var in ("BNG2_PL", "BNGPATH", "BioNetGenRoot"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(bngpath, "_bundled_bngpath", lambda: (None, "not installed"))
    real_which = shutil.which
    monkeypatch.setattr(
        bngpath.shutil,
        "which",
        lambda name, *a, **k: None if name == "BNG2.pl" else real_which(name, *a, **k),
    )
    bngpath._forget_perl_checks()
    yield
    bngpath._forget_perl_checks()


def _install(tmp_path: Path, body: str) -> Path:
    """A BioNetGen-shaped folder whose BNG2.pl is ``body``; returns the folder."""
    root = tmp_path / "bng"
    root.mkdir()
    (root / "BNG2.pl").write_text(body, encoding="utf-8")
    return root


COMPILES = "#!/usr/bin/env perl\nuse strict;\nuse FindBin;\n1;\n"  # what perl-base lacks
DOES_NOT = f"use strict;\nuse {MISSING};\n1;\n"


# ─── With the perl on this machine ──────────────────────────────────────────


@needs_perl
def test_a_perl_that_cannot_compile_bng2_pl_is_not_usable(tmp_path, monkeypatch):
    monkeypatch.setenv("BNGPATH", str(_install(tmp_path, DOES_NOT)))
    r = bngpath.resolve_bng()
    assert r.bng2_pl == tmp_path / "bng" / "BNG2.pl"  # found: this is not a miss
    assert r.has_perl is True  # a perl is there; it is the wrong one
    assert r.ok is False
    why = r.why_not()
    assert "cannot compile it" in why
    assert MISSING in why  # Perl's own line, so the reader sees which module
    assert "apt-get install perl" in why  # the fix for a module of Perl's own
    assert f"{MISSING} is not one of BioNetGen's own modules" in why  # as Perl spells it
    assert "probably part of Perl's own library" in why  # a guess, said as one
    assert "$BioNetGenRoot" not in why  # not the advice for one of BioNetGen's own
    assert "(@INC" not in why  # the search-path listing is dropped; "in @INC" stays
    assert bngpath._NOT_FOUND not in why  # the checkout footer must not fire for it
    assert bngpath.skip_reason() == why


@needs_perl
def test_a_perl_that_compiles_bng2_pl_is_usable(tmp_path, monkeypatch):
    monkeypatch.setenv("BNGPATH", str(_install(tmp_path, COMPILES)))
    r = bngpath.resolve_bng()
    assert r.ok is True
    assert r.has_perl is True
    assert r.perl_problem == ""
    assert r.why_not() == ""


@needs_perl
def test_has_bngl_capabilities_and_from_bngl_agree(tmp_path, monkeypatch):
    """The three callers of the resolver say the same thing (the issue's point:
    BNGL was reported available where the load failed)."""
    monkeypatch.setenv("BNGPATH", str(_install(tmp_path, DOES_NOT)))
    assert bngsim.HAS_BNGL is False
    caps = bngsim.capabilities()
    assert caps["features"]["bngl"] is False
    assert MISSING in caps["missing"]["bngl"]
    assert "apt-get install perl" in caps["missing"]["bngl"]
    model = tmp_path / "m.bngl"
    model.write_text("begin model\nend model\n", encoding="utf-8")
    with pytest.raises(ModelError, match="apt-get install perl") as info:
        bngsim.Model.from_bngl(str(model))
    assert MISSING in str(info.value)


@needs_perl
def test_the_probe_runs_perl_once_per_script_and_again_when_it_changes(tmp_path, monkeypatch):
    root = _install(tmp_path, COMPILES)
    monkeypatch.setenv("BNGPATH", str(root))
    calls: list[list[str]] = []
    real_run = subprocess.run

    def counting(args, *a, **k):
        calls.append(list(args))
        return real_run(args, *a, **k)

    monkeypatch.setattr(bngpath.subprocess, "run", counting)
    assert bngpath.resolve_bng().ok
    assert bngpath.resolve_bng().ok
    assert bngsim.capabilities()["features"]["bngl"] is True
    checks = [c for c in calls if c[1:2] == ["-c"]]  # capabilities() runs other things
    assert len(checks) == 1, checks  # one `perl -c` for three resolutions
    assert checks[0][2] == str(root / "BNG2.pl")

    # An edited script is compiled again. The mtime is pushed by a second, since
    # a filesystem may not record the difference between two quick writes.
    script = root / "BNG2.pl"
    script.write_text(DOES_NOT, encoding="utf-8")
    st = script.stat()
    os.utime(script, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert bngpath.resolve_bng().ok is False
    assert len([c for c in calls if c[1:2] == ["-c"]]) == 2


# BNG2.pl's own `use lib`: $BNGPATH, else $BioNetGenRoot, else its own folder,
# then "Perl2"; and one of BioNetGen's modules from there.
USES_ITS_LIBRARY = (
    "use strict;\nuse FindBin;\nuse File::Spec;\n"
    "use lib File::Spec->catdir(exists $ENV{'BNGPATH'} ? $ENV{'BNGPATH'}\n"
    "    : exists $ENV{'BioNetGenRoot'} ? $ENV{'BioNetGenRoot'} : $FindBin::RealBin, 'Perl2');\n"
    "use BNGUtils;\n1;\n"
)


def _install_with_library(tmp_path: Path) -> Path:
    root = _install(tmp_path, USES_ITS_LIBRARY)
    (root / "Perl2").mkdir()
    (root / "Perl2" / "BNGUtils.pm").write_text("package BNGUtils;\n1;\n", encoding="utf-8")
    return root


@needs_perl
@pytest.mark.parametrize("var", ["BNGPATH", "BioNetGenRoot"])
def test_a_library_variable_pointing_elsewhere_is_named_not_perl(tmp_path, monkeypatch, var):
    """One of BioNetGen's own modules missing means the library path is wrong, not
    Perl: the advice is the variable, not `apt-get install perl` (review of #1780)."""
    root = _install_with_library(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("BNG2_PL", str(root / "BNG2.pl"))  # found this way, whatever var says
    monkeypatch.setenv(var, str(elsewhere))
    r = bngpath.resolve_bng()
    assert r.bng2_pl == root / "BNG2.pl"
    assert r.ok is False
    assert "Can't locate BNGUtils.pm" in r.perl_problem
    why = r.why_not()
    assert "BNGUtils.pm is one of BioNetGen's own modules" in why
    assert "$BNGPATH or $BioNetGenRoot" in why
    assert "apt-get install perl" not in why


@needs_perl
def test_the_same_bng2_pl_with_its_own_library_compiles(tmp_path, monkeypatch):
    """Control: with neither variable set, its `use lib` finds Perl2 beside it."""
    root = _install_with_library(tmp_path)
    monkeypatch.setenv("BNG2_PL", str(root / "BNG2.pl"))
    r = bngpath.resolve_bng()
    assert r.ok is True, r.why_not()


symlinks_posix = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-specific: creating a symlink needs privileges"
)


@needs_perl
@symlinks_posix
def test_a_bng2_pl_reached_through_a_symlink_finds_its_own_modules(tmp_path, monkeypatch):
    """Conda puts a symlink to BNG2.pl on PATH, with no Perl2 folder beside the link.
    BNG2.pl finds its library from the real file ($FindBin::RealBin), so the check
    for one of its own modules must look there too (review of #1780)."""
    root = _install_with_library(tmp_path)
    on_path = tmp_path / "bin"
    on_path.mkdir()
    (on_path / "BNG2.pl").symlink_to(root / "BNG2.pl")
    monkeypatch.setenv("BNG2_PL", str(on_path / "BNG2.pl"))
    assert bngpath.resolve_bng().ok is True  # control: the link compiles, from RealBin

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("BioNetGenRoot", str(elsewhere))
    why = bngpath.resolve_bng().why_not()
    assert "BNGUtils.pm is one of BioNetGen's own modules" in why
    assert f"point them at {root.resolve()}" in why  # the real folder, not the link's
    assert "apt-get install perl" not in why


@needs_perl
def test_a_compile_error_that_names_no_module_gets_neither_diagnosis(tmp_path, monkeypatch):
    monkeypatch.setenv("BNGPATH", str(_install(tmp_path, "use strict;\nif (\n")))
    r = bngpath.resolve_bng()
    assert r.ok is False
    assert "syntax error" in r.perl_problem
    why = r.why_not()
    assert "check that BNG2.pl and its Perl2 library are complete" in why
    assert "apt-get install perl" not in why
    assert "$BioNetGenRoot" not in why


@needs_perl
def test_a_broken_locale_does_not_hide_the_missing_module(tmp_path, monkeypatch):
    """On a minimal image that forwards a LANG it has no locale for, perl warns
    before anything else; the message must still name the module."""
    monkeypatch.setenv("LC_ALL", "xx_YY.UTF-8")
    monkeypatch.setenv("BNGPATH", str(_install(tmp_path, DOES_NOT)))
    problem = bngpath.resolve_bng().perl_problem
    assert MISSING in problem
    assert "locale" not in problem


# ─── Without a perl: the text handling, with the image's own output ─────────


def _stub_run(monkeypatch, returncode: int, stderr: str = "", stdout: str = ""):
    seen: list[list[str]] = []

    def run(args, *a, **k):
        seen.append(list(args))
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(bngpath.subprocess, "run", run)
    monkeypatch.setattr(
        bngpath.shutil, "which", lambda n, *a, **k: "/usr/bin/perl" if n == "perl" else None
    )
    return seen


def test_the_message_relays_perls_first_line_without_the_inc_listing(tmp_path, monkeypatch):
    seen = _stub_run(monkeypatch, 2, stderr=SLIM_STDERR)
    monkeypatch.setenv("BNGPATH", str(_install(tmp_path, "# any\n")))
    r = bngpath.resolve_bng()
    assert seen == [["/usr/bin/perl", "-c", str(tmp_path / "bng" / "BNG2.pl")]]
    assert r.perl_problem == (
        "/usr/bin/perl said: Can't locate FindBin.pm in @INC "
        "(you may need to install the FindBin module)"
    )
    why = r.why_not()
    assert why.startswith(f"found BNG2.pl at {tmp_path / 'bng' / 'BNG2.pl'} but the perl on PATH")
    assert "FindBin.pm" in why
    assert "perl-base" in why
    assert "apt-get install perl" in why
    assert "entries checked" not in why and "BEGIN failed" not in why


LOCALE_PREAMBLE = (
    "perl: warning: Setting locale failed.\n"
    "perl: warning: Please check that your locale settings:\n"
    "\tLANGUAGE = (unset),\n"
    '\tLC_ALL = "xx_YY.UTF-8",\n'
    '\tLANG = "C.UTF-8"\n'
    "    are supported and installed on your system.\n"
    'perl: warning: Falling back to the standard locale ("C").\n'
)


@pytest.mark.parametrize(
    ("stderr", "line"),
    [
        (
            LOCALE_PREAMBLE + SLIM_STDERR,
            "Can't locate FindBin.pm in @INC (you may need to install the FindBin module)",
        ),
        (
            LOCALE_PREAMBLE + "syntax error at BNG2.pl line 2, at EOF\n",
            "syntax error at BNG2.pl line 2, at EOF",
        ),
        (
            "Use of uninitialized value in concatenation at BNG2.pl line 12.\n" + SLIM_STDERR,
            "Can't locate FindBin.pm in @INC (you may need to install the FindBin module)",
        ),
    ],
    ids=["missing-module", "syntax-error", "warning-before-module"],
)
def test_perls_locale_warning_does_not_take_the_place_of_its_line(
    tmp_path, monkeypatch, stderr, line
):
    _stub_run(monkeypatch, 2, stderr=stderr)
    monkeypatch.setenv("BNGPATH", str(_install(tmp_path, "# any\n")))
    assert bngpath.resolve_bng().perl_problem == f"/usr/bin/perl said: {line}"


def test_the_probe_runs_with_perls_locale_warning_off(tmp_path, monkeypatch):
    envs: list[dict] = []

    def run(args, *a, **k):
        envs.append(dict(k.get("env") or {}))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="syntax OK\n")

    monkeypatch.setattr(bngpath.subprocess, "run", run)
    monkeypatch.setattr(
        bngpath.shutil, "which", lambda n, *a, **k: "/usr/bin/perl" if n == "perl" else None
    )
    monkeypatch.setenv("LC_ALL", "xx_YY.UTF-8")
    monkeypatch.setenv("BNGPATH", str(_install(tmp_path, "# any\n")))
    assert bngpath.resolve_bng().ok
    assert envs[0]["PERL_BADLANG"] == "0"
    assert envs[0]["LC_ALL"] == "xx_YY.UTF-8"  # otherwise BNG2.pl's own environment


def test_a_failure_with_no_output_reports_the_exit_status(tmp_path, monkeypatch):
    _stub_run(monkeypatch, 255)
    monkeypatch.setenv("BNGPATH", str(_install(tmp_path, "# any\n")))
    assert bngpath.resolve_bng().perl_problem == "/usr/bin/perl said: exit status 255"


def test_a_probe_that_times_out_is_reported_as_unusable(tmp_path, monkeypatch):
    def run(args, *a, **k):
        raise subprocess.TimeoutExpired(args, k.get("timeout", 0))

    monkeypatch.setattr(bngpath.subprocess, "run", run)
    monkeypatch.setattr(
        bngpath.shutil, "which", lambda n, *a, **k: "/usr/bin/perl" if n == "perl" else None
    )
    monkeypatch.setenv("BNGPATH", str(_install(tmp_path, "# any\n")))
    r = bngpath.resolve_bng()
    assert r.ok is False
    why = r.why_not()
    assert "did not finish within" in why
    assert "could not compile it" in why
    assert "apt-get install perl" not in why  # no library is missing; the perl itself is


def test_no_perl_at_all_is_still_the_stock_windows_message(tmp_path, monkeypatch):
    """Control: the pre-existing refusal, which a compile probe must not reach."""
    monkeypatch.setattr(bngpath.shutil, "which", lambda n, *a, **k: None)
    monkeypatch.setenv("BNGPATH", str(_install(tmp_path, "# any\n")))
    r = bngpath.resolve_bng()
    assert r.ok is False and r.has_perl is False and r.perl_problem == ""
    assert "`perl` is not on PATH" in r.why_not()
    assert "apt-get install perl" not in r.why_not()


def test_nothing_found_does_not_run_perl(monkeypatch):
    """With no BNG2.pl there is nothing to compile; has_perl stays a PATH check."""
    seen = _stub_run(monkeypatch, 2, stderr="must not run")
    r = bngpath.resolve_bng()
    assert r.bng2_pl is None and r.has_perl is True and r.perl_problem == ""
    assert seen == []
    assert bngpath._NOT_FOUND in r.why_not()


# ─── A fake perl first on PATH (POSIX): what from_bngl runs, and in which env ─
#
# A /bin/sh script that answers `-c` as perl-base did or as a full Perl does, and
# logs each call with the $BNGPATH it saw. BNG2.pl is a file of that name given
# through $BNGPATH; the fake never reads it.

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-specific: the fake perl is a /bin/sh script"
)

FINDBIN = "Can't locate FindBin.pm in @INC (you may need to install the FindBin module)"
OLDER_PERL_STDERR = (
    FINDBIN + " (@INC contains: /etc/perl /usr/share/perl5) at BNG2.pl line 34.\n"
    "BEGIN failed--compilation aborted at BNG2.pl line 34.\n"
)


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A `bin/perl` first on PATH and a stand-in BNG2.pl under $BNGPATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    root = _install(tmp_path, "# stand-in; the fake perl never reads it\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("BNGPATH", str(root))
    return SimpleNamespace(
        bin=bin_dir, bng2=root / "BNG2.pl", root=root, log=tmp_path / "perl.log"
    )


def _install_fake_perl(fake: SimpleNamespace, *, exit_code: int, stderr: str) -> None:
    script = fake.bin / "perl"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s|%s\\n" "$*" "$BNGPATH" >> {shlex.quote(str(fake.log))}\n'
        f"printf '%s' {shlex.quote(stderr)} >&2\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


def _calls(fake: SimpleNamespace) -> list[str]:
    return fake.log.read_text(encoding="utf-8").splitlines() if fake.log.exists() else []


@posix_only
def test_from_bngl_refuses_before_running_bng2_pl(fake, tmp_path):
    _install_fake_perl(fake, exit_code=2, stderr=SLIM_STDERR)
    model = tmp_path / "m.bngl"
    model.write_text("begin model\nend model\n", encoding="utf-8")
    with pytest.raises(ModelError, match="cannot load BNGL") as info:
        bngsim.Model.from_bngl(str(model))
    assert FINDBIN in str(info.value)
    assert "apt-get install perl" in str(info.value)
    # Only the compile check ran: BNG2.pl itself was never started on the model.
    assert [call.split("|")[0] for call in _calls(fake)] == [f"-c {fake.bng2}"]


@posix_only
def test_the_older_inc_contains_form_is_cut_the_same_way(fake):
    _install_fake_perl(fake, exit_code=2, stderr=OLDER_PERL_STDERR)
    r = bngpath.resolve_bng()
    assert r.has_perl is True and r.ok is False
    assert r.perl_problem == f"{fake.bin / 'perl'} said: {FINDBIN}"


@posix_only
def test_the_probe_runs_in_bng2_pls_own_environment(fake, monkeypatch, tmp_path):
    # BNG2.pl runs with a relative $BNGPATH made absolute against the process's
    # directory (issue #1494); the probe must compile under the same Perl2 library.
    _install_fake_perl(fake, exit_code=0, stderr="syntax OK\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BNGPATH", "bng")
    assert bngpath.resolve_bng().ok
    assert _calls(fake) == [f"-c {fake.bng2}|{tmp_path / 'bng'}"]


@posix_only
def test_a_changed_library_environment_is_probed_again(fake, monkeypatch):
    _install_fake_perl(fake, exit_code=0, stderr="syntax OK\n")
    bngpath.resolve_bng()
    bngpath.resolve_bng()
    assert len(_calls(fake)) == 1
    # The same script under a $BioNetGenRoot that swaps its Perl2 modules.
    monkeypatch.setenv("BioNetGenRoot", str(fake.root))
    bngpath.resolve_bng()
    assert len(_calls(fake)) == 2
    assert _calls(fake)[1].endswith(f"|{fake.root}")  # and BNGPATH still reached it


# ─── Fixed in place, and a process whose directory is gone ──────────────────


@needs_perl
def test_a_refusal_is_asked_again_so_a_perl_fixed_in_place_is_seen(tmp_path, monkeypatch):
    """`apt-get install perl` changes neither perl's path nor BNG2.pl's mtime, so a
    remembered refusal would outlive the fix; only successes are remembered."""
    root = _install(tmp_path, DOES_NOT)
    monkeypatch.setenv("BNGPATH", str(root))
    script = root / "BNG2.pl"
    before = script.stat()
    assert bngpath.resolve_bng().ok is False
    # The same perl, the same path, the same mtime: only the script's content
    # (standing in for the perl's library) changes.
    script.write_text(COMPILES, encoding="utf-8")
    os.utime(script, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert script.stat().st_mtime_ns == before.st_mtime_ns
    assert bngpath.resolve_bng().ok is True


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-specific: Windows cannot remove a process's working directory",
)
def test_the_probe_answers_when_the_working_directory_is_gone(tmp_path, monkeypatch):
    """The environment BNG2.pl runs in reads the working directory only for a
    relative $BNGPATH; a process whose directory was removed still gets an answer
    from resolve_bng(), HAS_BNGL and capabilities() (review of #1761)."""
    seen = _stub_run(monkeypatch, 0, stderr="syntax OK\n")
    monkeypatch.setenv("BNGPATH", str(_install(tmp_path, "# any\n")))
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    r = bngpath.resolve_bng()
    assert r.ok is True
    assert len(seen) == 1
    assert bngsim.HAS_BNGL is True
    assert bngsim.capabilities()["features"]["bngl"] is True
