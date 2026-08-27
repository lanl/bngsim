"""``gen_networks.py`` must create the ``.net`` cache it exists to fill.

Issue #492. Phase 1 of the full-network ODE benchmark writes every generated
network into ``NETS = HERE / "nets"`` and never created that directory. On a
machine where it did not exist -- a fresh checkout, or one where the cache had
been cleaned to reclaim disk -- every model failed. Not cheaply, either: the
generation work happens first and only the *write* fails, so six models burned
28 s to produce nothing, and the full corpus would burn hours.

The shape of it is what makes it worth a test rather than a one-line commit.
The script whose whole job is to *build* the cache was the one script that
could not create it, so the documented recovery path for a missing cache did
not work -- and the paper's Figures 1 and 3 and Tables S3/S4 all read that
cache. A missing directory is exactly the state you are in when you most need
this script.

The failure also named the wrong thing. ``FileNotFoundError`` on a ``.net``
path reads as "this model could not be generated", which is what a genuine
BNG2.pl failure looks like too, and the console truncates a row's detail, so
the path the reader needed was cut off. ``ensure_cache_dir`` makes that
unreachable on a normal run; ``describe_write_failure`` covers the case it
cannot -- the directory removed *while* a multi-hour sweep is in flight -- and
says so in those words instead of borrowing netgen's error shape.

BNG2.pl is not needed here and is deliberately not used: these tests drive the
argv path that stops before any generation, which is the whole point (the
directory has to exist *before* the work, not after it).
"""

from __future__ import annotations

import errno
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "benchmarks" / "suites" / "ode_fullnet" / "gen_networks.py"

pytestmark = pytest.mark.skipif(
    not GEN.exists(),
    reason="benchmarks/ is not in this checkout (installed package)",
)


@pytest.fixture(scope="module")
def gen_networks():
    """Import ``gen_networks.py`` by path, leaving ``sys.path`` as we found it.

    The module puts ``parity_checks/`` and ``parity_checks/bng_parity`` on
    ``sys.path`` at import so it can reach ``_bng_common``; that is its business,
    not the rest of the session's.
    """
    saved_path = list(sys.path)
    name = "gen_networks_under_test"
    spec = importlib.util.spec_from_file_location(name, GEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.path[:] = saved_path
        sys.modules.pop(name, None)


def test_ensure_cache_dir_creates_a_missing_cache(gen_networks, tmp_path):
    nets = tmp_path / "fresh-checkout" / "nets"
    assert not nets.exists()
    gen_networks.ensure_cache_dir(nets)
    assert nets.is_dir()


def test_ensure_cache_dir_is_idempotent(gen_networks, tmp_path):
    """It runs at the head of every pass, including the 585-network resume."""
    nets = tmp_path / "nets"
    gen_networks.ensure_cache_dir(nets)
    (nets / "already_here.net").write_text("x")
    gen_networks.ensure_cache_dir(nets)
    assert (nets / "already_here.net").read_text() == "x"


def test_a_pass_creates_the_cache_before_doing_any_work(gen_networks, tmp_path, monkeypatch):
    """The directory exists by the time the run reports it, not after the writes.

    Driven through ``main()`` with a model filter that matches nothing, so the
    run reaches its "nothing to do" exit without BNG2.pl and without generating
    anything -- which is precisely the ordering under test: #492 was not that
    the cache was never created, it was that it was not created *first*.
    """
    nets = tmp_path / "nets"
    monkeypatch.setattr(gen_networks, "NETS", nets)
    monkeypatch.setattr(gen_networks, "resolve_bng2pl", lambda: "/nonexistent/BNG2.pl")

    rc = gen_networks.main(["--models", "no-model-has-this-substring"])

    assert rc == 0
    assert nets.is_dir(), "the pass ran without creating the cache it writes into"


def test_a_vanished_cache_directory_names_itself(gen_networks, tmp_path):
    """The residual case: removed mid-sweep, after ensure_cache_dir ran."""
    nets = tmp_path / "nets"  # deliberately absent
    dest = nets / "some_model.net"
    exc = FileNotFoundError(errno.ENOENT, "No such file or directory", str(dest))

    msg = gen_networks.describe_write_failure(exc, dest, nets)

    assert str(nets) in msg
    assert "cache directory" in msg
    # The console truncates a row's detail, so the directory has to come first.
    assert msg.startswith(str(nets))


def test_a_real_write_failure_is_not_mislabelled(gen_networks, tmp_path):
    """A present cache directory means the ENOENT is about something else."""
    nets = tmp_path / "nets"
    nets.mkdir()
    dest = nets / "some_model.net"
    exc = FileNotFoundError(errno.ENOENT, "No such file or directory", str(dest))

    msg = gen_networks.describe_write_failure(exc, dest, nets)

    assert "cache directory" not in msg
    assert str(dest) in msg


def test_a_non_enoent_failure_is_not_mislabelled(gen_networks, tmp_path):
    """A full disk or a read-only mount is not a missing directory."""
    nets = tmp_path / "nets"  # absent, so only the errno separates the two cases
    dest = nets / "some_model.net"
    exc = PermissionError(errno.EACCES, "Permission denied", str(dest))

    msg = gen_networks.describe_write_failure(exc, dest, nets)

    assert "cache directory" not in msg
    assert "PermissionError" in msg
