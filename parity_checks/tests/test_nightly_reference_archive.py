"""The nightly's reference archive plumbing (issue #805): pack, manifest, fetch.

``parity_checks/nightly/reference_archive.py`` packs what a record-mode night
recorded into one asset per suite, writes the manifest the repository commits,
and fetches an asset back, checked against that manifest, for a replay night.
These tests drive it on synthetic entries; no runner, engine or network is used
(the asset is fetched through a ``file://`` URL).
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import reference_archive as ra

_PC = Path(__file__).resolve().parent.parent
_WORKFLOW = _PC.parent / ".github" / "workflows" / "nightly-parity.yml"

# One CC0 BioModel, one MIT RuleHub model, and a model no provenance manifest knows.
_CC0 = "BIOMD0000000001"
_MIT = "fast/rulehub/Examples/biology/aktsignaling/akt-signaling.bngl"
_UNKNOWN = "BIOMD9999999999"


def _entry(d: Path, model_id: str, method: str | None = "ode", tag: str = "0") -> Path:
    meta = {"model_id": model_id, "reference_version": "2.9.3"}
    if method:
        meta["method"] = method
    name = re.sub(r"[^A-Za-z0-9._-]", "_", f"{model_id}:{method}") + f"-{tag * 12}.npz"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / name,
        meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
        time=np.linspace(0.0, 1.0, 5),
    )
    return d / name


def _report(path: Path, *rows: tuple[str, str, str]) -> Path:
    path.write_text(
        json.dumps({"results": [{"model_id": m, "method": k, "outcome": o} for m, k, o in rows]})
    )
    return path


def _recorded(tmp_path: Path) -> Path:
    src = tmp_path / "recorded"
    _entry(src, _CC0, tag="a")
    _entry(src, _MIT, tag="b")
    _entry(src, _UNKNOWN, tag="c")
    return src


def test_pack_leaves_out_what_no_license_allows_and_is_deterministic(tmp_path):
    src = _recorded(tmp_path)
    one = ra.pack("bng_ode", src, tmp_path / "one")
    two = ra.pack("bng_ode", src, tmp_path / "two")
    assert one["sha256"] == two["sha256"]  # same bytes, whatever the clock says
    assert one["n_entries"] == 2
    assert [(x["model_id"], x["reason"]) for x in one["excluded"]] == [
        (_UNKNOWN, "license unknown")
    ]
    assert one["licenses"] == ["CC0-1.0", "MIT"]
    with tarfile.open(tmp_path / "one" / "reference-bng_ode.tar.gz") as tar:
        names = tar.getnames()
        entries = json.loads(tar.extractfile("reference-bng_ode/entries.json").read())
    # An asset holding a non-CC0 entry carries the notices its licenses require.
    assert "reference-bng_ode/NOTICE" in names
    assert "reference-bng_ode/third_party/rulemonkey/LICENSE" in names
    assert {e["model_id"]: e["license"] for e in entries} == {_CC0: "CC0-1.0", _MIT: "MIT"}


def test_an_all_cc0_asset_needs_no_notice(tmp_path):
    src = tmp_path / "recorded"
    _entry(src, _CC0)
    side = ra.pack("rr_ode", src, tmp_path / "out")
    with tarfile.open(tmp_path / "out" / side["asset"]) as tar:
        assert not any(n.endswith("NOTICE") for n in tar.getnames())


def test_pack_refuses_when_a_good_row_has_no_recorded_reference(tmp_path):
    """Published, such an archive would turn that PASS into an archive_stale alert
    every night until someone re-recorded it."""
    src = _recorded(tmp_path)
    report = _report(
        tmp_path / "report.json",
        (_CC0, "ode", "PASS"),
        ("BIOMD0000000002", "ode", "PASS"),
        ("BIOMD0000000003", "ode", "TIMEOUT"),  # killed before its reference: listed
    )
    with pytest.raises(SystemExit, match=r"1 PASS row.*BIOMD0000000002\|ode"):
        ra.pack("rr_ode", src, tmp_path / "out", [report])


def test_pack_lists_the_timeouts_the_recording_left_without_a_reference(tmp_path):
    """Issue #1290: three jobs timed out in the 2026-10-24 recording and left no
    entry, and the night after a live night that passed them alerted on all three
    as archive_stale. Nothing said so when the archive was published; the publish
    summary now does. A row of another outcome may need no entry (ssa_screen
    decides its time-event rows before it reads the archive), so only TIMEOUT
    rows are listed."""
    src = _recorded(tmp_path)
    report = _report(
        tmp_path / "report.json",
        (_CC0, "ode", "PASS"),
        (_MIT, "ode", "TIMEOUT"),  # an entry, saved before the kill: not listed
        ("BIOMD0000000003", "ode", "TIMEOUT"),
        ("BIOMD0000000004", "ssa", "DIFF"),  # a time-event row: no entry needed
    )
    giants = _report(tmp_path / "report-1.json", ("MODEL1504130000", "ode", "TIMEOUT"))
    side = ra.pack("rr_ode", src, tmp_path / "out", [report, giants])
    assert side["timed_out_unrecorded"] == ["BIOMD0000000003|ode", "MODEL1504130000|ode"]
    for suite in ra.SUITES:
        if suite != "rr_ode":
            ra.pack(suite, _recorded(tmp_path / suite), tmp_path / "out")
    (line,) = ra.caveats(tmp_path / "out")
    assert line == (
        "- **rr_ode**: 2 case(s) timed out before their reference was recorded; a replay "
        "that needs it refuses them as archive_stale: BIOMD0000000003|ode, MODEL1504130000|ode"
    )


def test_pack_lists_baseline_good_cases_the_recording_did_not_pass(tmp_path):
    """Such a case was recorded down whatever path it took (an SSA row that timed out
    into a partial-horizon ladder records only that ladder), so a replay may refuse
    it; it is listed for the publish summary, not refused."""
    src = _recorded(tmp_path)
    report = _report(
        tmp_path / "report.json", (_CC0, "ode", "PASS"), ("BIOMD0000000002", "ode", "DIFF")
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "cases": {
                    f"{_CC0}|ode": {"outcome": "PASS"},
                    "BIOMD0000000002|ode": {"outcome": "PASS"},
                    "BIOMD0000000003|ode": {"outcome": "TIMEOUT"},
                }
            }
        )
    )
    side = ra.pack("rr_ode", src, tmp_path / "out", [report], baseline)
    assert side["good_in_baseline_only"] == ["BIOMD0000000002|ode"]


def test_an_entry_naming_no_method_covers_every_row_of_its_model(tmp_path):
    src = tmp_path / "recorded"
    _entry(src, _CC0, method=None)  # bng_stoch's and ssa_screen's entries
    report = _report(tmp_path / "report.json", (_CC0, "ssa", "PASS"))
    assert ra.pack("rr_ssa", src, tmp_path / "out", [report])["n_entries"] == 1


def _published(tmp_path: Path, suite: str = "bng_ode") -> Path:
    """Pack every suite (the manifest wants all or none), and point the manifest at
    the packed files through file:// URLs instead of the release."""
    out = tmp_path / "assets"
    for s in ra.SUITES:
        ra.pack(s, _recorded(tmp_path / s), out)
    doc = ra.manifest("reference-data-v1", out, tmp_path / "manifest.json", today="2026-10-24")
    for rec in doc["records"]:
        rec["origin"]["url"] = (out / f"reference-{rec['id']}.tar.gz").as_uri()
    (tmp_path / "manifest.json").write_text(json.dumps(doc))
    return tmp_path / "manifest.json"


def test_the_manifest_conforms_to_the_corpus_manifest_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    doc = json.loads(_published(tmp_path).read_text())
    schema = json.loads((_PC / "manifest.schema.json").read_text())
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(doc))
    assert not errors, errors[:3]
    rec = {r["id"]: r for r in doc["records"]}
    assert rec["bng_ode"]["license"] == {"spdx": "CC0-1.0 AND MIT", "notice": "NOTICE"}
    assert (rec["bng_ode"]["n_entries"], rec["bng_ode"]["n_excluded"]) == (2, 1)
    assert rec["bng_ode"]["origin"]["url"].endswith("reference-bng_ode.tar.gz")


def test_the_manifest_wants_every_suite_or_none(tmp_path):
    out = tmp_path / "assets"
    ra.pack("rr_ode", _recorded(tmp_path), out)
    with pytest.raises(SystemExit, match="every suite or none"):
        ra.manifest("reference-data-v1", out, tmp_path / "manifest.json")


def test_fetch_checks_the_pinned_sha256_and_unpacks_only_entries(tmp_path):
    manifest = _published(tmp_path)
    dest = tmp_path / "replay"
    assert ra.fetch("bng_ode", dest, manifest) == 2
    recorded = (tmp_path / "bng_ode" / "recorded").glob("*.npz")
    want = sorted(p.name for p in recorded if not p.name.startswith(_UNKNOWN))
    assert sorted(p.name for p in dest.iterdir()) == want  # entries only, no README or NOTICE
    doc = json.loads(manifest.read_text())
    for rec in doc["records"]:
        rec["sha256"] = "0" * 64
    manifest.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="the manifest pins 0000"):
        ra.fetch("bng_ode", tmp_path / "again", manifest)


def test_fetch_writes_no_member_outside_its_own_entries(tmp_path):
    """A crafted asset (a path out of dest, a name no runner writes) yields only
    the well-formed entry, and the count check then refuses the shortfall."""
    asset = tmp_path / "evil.tar.gz"
    good = "reference-rr_ode/BIOMD1_ode-0123456789ab.npz"
    with tarfile.open(asset, "w:gz") as tar:
        for name in (good, "reference-rr_ode/../../escaped.npz", "reference-rr_ode/run.sh"):
            info = tarfile.TarInfo(name)
            info.size = 3
            tar.addfile(info, io.BytesIO(b"npz"))
    manifest = tmp_path / "manifest.json"
    rec = {
        "id": "rr_ode",
        "origin": {"kind": "url", "url": asset.as_uri()},
        "sha256": ra._sha256_file(asset),
        "n_entries": 2,
    }
    manifest.write_text(json.dumps({"records": [rec]}))
    with pytest.raises(SystemExit, match="1 entries unpacked; the manifest records 2"):
        ra.fetch("rr_ode", tmp_path / "dest", manifest)
    assert [p.name for p in (tmp_path / "dest").iterdir()] == ["BIOMD1_ode-0123456789ab.npz"]
    assert not (tmp_path / "escaped.npz").exists()


def _no_download(monkeypatch):
    def refuse(url, tmp):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(ra, "_download", refuse)


def test_locate_names_the_pinned_asset(tmp_path):
    """What .github/actions/reference-side hands cached-download (#1607)."""
    manifest = _published(tmp_path)
    rec = {r["id"]: r for r in json.loads(manifest.read_text())["records"]}["bng_ode"]
    assert ra.locate("bng_ode", manifest) == [
        f"url={rec['origin']['url']}",
        f"sha256={rec['sha256']}",
        "name=reference-bng_ode.tar.gz",
    ]


def test_fetch_reads_a_local_copy_checked_like_a_download(tmp_path, monkeypatch):
    """A copy of the asset is unpacked with no download, and only if its sha256
    is the one the manifest pins (#1607)."""
    manifest = _published(tmp_path)
    _no_download(monkeypatch)
    asset = tmp_path / "assets" / "reference-bng_ode.tar.gz"
    assert ra.fetch("bng_ode", tmp_path / "replay", manifest, archive=asset) == 2
    other = tmp_path / "other.tar.gz"
    other.write_bytes(asset.read_bytes() + b"\0")
    with pytest.raises(
        SystemExit, match=f"{re.escape(str(other))} has sha256 .*the manifest pins"
    ):
        ra.fetch("bng_ode", tmp_path / "again", manifest, archive=other)
    assert not (tmp_path / "again").exists()
    with pytest.raises(SystemExit, match="is not a file"):
        ra.fetch("bng_ode", tmp_path / "again", manifest, archive=tmp_path / "absent.tar.gz")


def test_the_command_line_locates_and_replays_from_a_local_copy(tmp_path):
    """The two calls reference-side makes, as it makes them."""
    manifest = _published(tmp_path)
    script = Path(ra.__file__)

    def run(*args):
        done = subprocess.run(
            [sys.executable, str(script), *args, "--manifest", str(manifest)],
            capture_output=True,
            text=True,
            check=True,
        )
        return dict(line.split("=", 1) for line in done.stdout.split("\n") if line)

    pin = run("locate", "bng_ode")
    assert pin["name"] == "reference-bng_ode.tar.gz"
    copy = tmp_path / "download" / pin["name"]
    copy.parent.mkdir()
    copy.write_bytes((tmp_path / "assets" / pin["name"]).read_bytes())
    # Downloading is impossible from here on, so a replay that tried would fail.
    (tmp_path / "assets" / pin["name"]).unlink()
    env = run(
        "prepare",
        "bng_ode",
        "--mode",
        "replay",
        "--root",
        str(tmp_path / "ref"),
        "--archive",
        str(copy),
    )
    assert env["ARCHIVE_DIR"] == str(tmp_path / "ref" / "bng_ode")
    assert len(list((tmp_path / "ref" / "bng_ode").glob("*.npz"))) == 2


def test_fetch_refuses_a_suite_the_manifest_does_not_hold(tmp_path):
    with pytest.raises(SystemExit, match="no reference archive manifest"):
        ra.fetch("rr_ode", tmp_path / "d", tmp_path / "absent.json")


@pytest.mark.parametrize(
    ("mode", "flag"),
    [("live", ""), ("record", "--write-reference-archive"), ("replay", "--reference-archive")],
)
def test_prepare_hands_each_mode_its_runner_flags(tmp_path, mode, flag):
    manifest = _published(tmp_path) if mode == "replay" else tmp_path / "unused.json"
    lines = dict(
        line.split("=", 1) for line in ra.prepare("bng_ode", mode, tmp_path / "ref", manifest)
    )
    d = tmp_path / "ref" / "bng_ode"
    assert lines["ARCHIVE_FLAGS"] == (f"{flag} {d}" if flag else "")
    assert lines["ARCHIVE_DIR"] == (str(d) if flag else "")
    if mode != "live":
        assert d.is_dir()


def test_the_workflow_prepares_every_suite_and_plans_from_its_own_schedules():
    """Every suite the script knows is prepared, packed and uploaded by the
    nightly, and the cron string the plan job tests is one the workflow has."""
    text = _WORKFLOW.read_text()
    # Each replay leg prepares through .github/actions/reference-side (#1607).
    prepared = set(
        re.findall(r"uses: \./\.github/actions/reference-side\n\s+with:\n\s+suite: (\S+)", text)
    )
    packed = set(re.findall(r"reference_archive\.py pack (\S+)", text))
    expand = {"bng_${{": {"bng_ssa", "bng_nf"}}
    norm = lambda names: set().union(*(expand.get(n, {n}) for n in names))  # noqa: E731
    assert norm(prepared) == set(ra.SUITES) == norm(packed)
    planned = re.search(r'if \[ "\$SCHEDULE" = "([^"]+)" \]', text).group(1)
    assert f'- cron: "{planned}"' in text
