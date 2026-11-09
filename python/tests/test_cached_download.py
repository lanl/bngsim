"""ci/cached_download.py leaves the pinned file at its path, or fails (#1607).

.github/actions/cached-download restores the file from the Actions cache and
then runs this script. On a hit the script checks the copy and downloads
nothing. On a miss it downloads, retrying, and the action saves the result.
Each test pins one rule:

* a copy with the pinned sha256 is kept, and nothing is downloaded;
* a copy with other bytes is replaced by a fresh download;
* a missing copy is downloaded and checked;
* a download with other bytes fails, and leaves nothing at the path;
* a request that fails is retried, and a run of failures longer than the
  retries fails with nothing left behind;
* the action reads ``downloaded=`` from $GITHUB_OUTPUT, and a malformed sha256
  is refused before anything is fetched.
"""

from __future__ import annotations

import hashlib
import http.server
import importlib.util
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402

REPO_ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "ci" / "cached_download.py"

if not SCRIPT.is_file():  # an installed wheel carries no ci/ directory
    pytest.skip("ci/cached_download.py lives in the source root", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("bngsim_ci_cached_download", SCRIPT)
cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cd)

PAYLOAD = b"the pinned archive\n" * 64
SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
NO_WAIT = (0, 0, 0, 0)


@pytest.fixture
def server():
    """A local server that answers 504 to its first ``fail`` requests, then
    serves ``body``. Counts the requests it saw."""

    class State:
        fail = 0
        body = PAYLOAD
        requests = 0

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            State.requests += 1
            if State.requests <= State.fail:
                self.send_error(504, "Gateway Time-out")
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(State.body)))
            self.end_headers()
            self.wfile.write(State.body)

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    State.url = f"http://127.0.0.1:{httpd.server_address[1]}/sundials-7.2.1.tar.gz"
    yield State
    httpd.shutdown()
    thread.join()


def _left(dest: Path) -> list[str]:
    return sorted(p.name for p in dest.parent.iterdir())


def test_a_copy_with_the_pinned_sha256_is_kept_and_nothing_is_downloaded(tmp_path, server):
    dest = tmp_path / "sundials-7.2.1.tar.gz"
    dest.write_bytes(PAYLOAD)
    assert cd.ensure(server.url, SHA256, dest, NO_WAIT) is False
    assert server.requests == 0
    assert dest.read_bytes() == PAYLOAD


def test_a_copy_with_other_bytes_is_downloaded_again(tmp_path, server):
    dest = tmp_path / "sundials-7.2.1.tar.gz"
    dest.write_bytes(PAYLOAD + b"\0")
    assert cd.ensure(server.url, SHA256, dest, NO_WAIT) is True
    assert server.requests == 1
    assert dest.read_bytes() == PAYLOAD
    assert _left(dest) == [dest.name]


def test_a_missing_copy_is_downloaded_and_checked(tmp_path, server):
    dest = tmp_path / "cache" / "sundials-7.2.1.tar.gz"
    assert cd.ensure(server.url, SHA256, dest, NO_WAIT) is True
    assert dest.read_bytes() == PAYLOAD
    assert _left(dest) == [dest.name]


def test_a_download_with_other_bytes_fails_and_leaves_nothing(tmp_path, server):
    server.body = PAYLOAD + b"tampered"
    dest = tmp_path / "sundials-7.2.1.tar.gz"
    with pytest.raises(SystemExit, match=f"has sha256 .*expected {SHA256}"):
        cd.ensure(server.url, SHA256, dest, NO_WAIT)
    assert _left(dest) == []


def test_failed_requests_are_retried(tmp_path, server):
    server.fail = 4  # the last of five attempts succeeds
    dest = tmp_path / "sundials-7.2.1.tar.gz"
    assert cd.ensure(server.url, SHA256, dest, NO_WAIT) is True
    assert server.requests == 5
    assert dest.read_bytes() == PAYLOAD


def test_more_failures_than_retries_fail_with_nothing_left(tmp_path, server):
    server.fail = 5
    dest = tmp_path / "sundials-7.2.1.tar.gz"
    with pytest.raises(SystemExit, match="could not download"):
        cd.ensure(server.url, SHA256, dest, NO_WAIT)
    assert server.requests == 5
    assert _left(dest) == []


def test_the_retries_wait_and_stop_at_the_last_attempt(tmp_path, server, monkeypatch):
    waited = []
    monkeypatch.setattr(cd.time, "sleep", waited.append)
    server.fail = 99
    with pytest.raises(SystemExit):
        cd.ensure(server.url, SHA256, tmp_path / "x.tar.gz")
    assert waited == list(cd.RETRY_DELAYS)
    assert server.requests == len(cd.RETRY_DELAYS) + 1


def _main(
    tmp_path: Path, url: str, sha256: str, dest: Path
) -> tuple[subprocess.CompletedProcess, str]:
    out = tmp_path / "github_output"
    out.write_text("")
    env = {**os.environ, "GITHUB_OUTPUT": str(out)}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--url", url, "--sha256", sha256, "--dest", str(dest)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return proc, out.read_text()


def test_the_action_reads_whether_it_downloaded(tmp_path, server):
    dest = tmp_path / "sundials-7.2.1.tar.gz"
    proc, out = _main(tmp_path, server.url, SHA256, dest)
    assert proc.returncode == 0, proc.stderr
    assert out == "downloaded=true\n"
    proc, out = _main(tmp_path, server.url, SHA256.upper(), dest)
    assert proc.returncode == 0, proc.stderr
    assert out == "downloaded=false\n"
    assert server.requests == 1


def test_a_malformed_sha256_is_refused_before_any_request(tmp_path, server):
    proc, out = _main(tmp_path, server.url, "deadbeef", tmp_path / "x.tar.gz")
    assert proc.returncode != 0
    assert "is not a sha256 hex digest" in proc.stderr
    assert out == ""
    assert server.requests == 0
