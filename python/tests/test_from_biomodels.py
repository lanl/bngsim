"""Model.from_biomodels fetches a BioModels entry by id and keeps a copy (#1706).

The tests replay recorded BioModels responses, so they need no network: CI
runners can't reach BioModels anyway (#807). A live check was made by hand
against BIOMD0000000010.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error

import bngsim
import pytest
from bngsim import ModelError, _biomodels

SBML = bngsim.examples.path("decay", "sbml").read_bytes()
SHA = hashlib.sha256(SBML).hexdigest()
ID = "BIOMD0000000010"
FILE = f"{ID}_url.xml"


def _meta(sha256: str = SHA, versions=(1, 2, 4)) -> bytes:
    return json.dumps(
        {
            "files": {"main": [{"name": FILE, "sha256sum": sha256}]},
            "history": {"revisions": [{"version": v} for v in versions]},
        }
    ).encode()


class FakeBioModels:
    """Answers the two requests from_biomodels makes, and counts them."""

    def __init__(self, *, meta: bytes | None = None, data: bytes = SBML, offline=False):
        self.meta = _meta() if meta is None else meta
        self.data = data
        self.offline = offline
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        if self.offline:
            raise urllib.error.URLError("no network")
        if "?format=json" in url:
            entry = url.split("/")[-1].split("?")[0]
            if not entry.startswith(ID):
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
            return self.meta
        return self.data


@pytest.fixture
def biomodels(monkeypatch, tmp_path):
    fake = FakeBioModels()
    monkeypatch.setattr(_biomodels, "_get", fake)
    monkeypatch.setattr(_biomodels, "CACHE_DIR", tmp_path)
    return fake


def test_it_fetches_the_main_file_and_records_where_it_came_from(biomodels, tmp_path):
    model = bngsim.Model.from_biomodels(ID)
    assert model.species_names == ["A"]
    assert biomodels.urls == [
        f"https://www.biomodels.org/{ID}?format=json",
        f"https://www.biomodels.org/model/download/{ID}.4?filename={FILE}",
    ]
    source = model.describe()["source"]
    assert source["biomodels_id"] == ID
    assert source["biomodels_version"] == 4
    assert source["biomodels_file"] == FILE
    assert source["sha256"] == SHA
    assert (tmp_path / f"{SHA}.xml").read_bytes() == SBML


def test_a_second_call_needs_no_network(biomodels):
    bngsim.Model.from_biomodels(ID)
    biomodels.offline = True
    model = bngsim.Model.from_biomodels(ID)
    assert model.species_names == ["A"]
    assert len(biomodels.urls) == 2  # the first call's two requests only


def test_a_version_pins_the_file(biomodels):
    bngsim.Model.from_biomodels(ID, version=2)
    assert biomodels.urls[0] == f"https://www.biomodels.org/{ID}.2?format=json"
    assert biomodels.urls[1].startswith(f"https://www.biomodels.org/model/download/{ID}.2?")
    biomodels.offline = True
    bngsim.Model.from_biomodels(ID, version=2)  # cached by version
    with pytest.raises(ModelError, match="No copy is cached"):
        bngsim.Model.from_biomodels(ID, version=1)


def test_cache_false_fetches_again_and_keeps_nothing(biomodels, tmp_path):
    model = bngsim.Model.from_biomodels(ID, cache=False)
    assert model.describe()["source"]["sha256"] == SHA
    assert list(tmp_path.iterdir()) == []


def test_a_damaged_cached_file_is_fetched_again(biomodels, tmp_path):
    bngsim.Model.from_biomodels(ID)
    (tmp_path / f"{SHA}.xml").write_bytes(b"damaged")
    bngsim.Model.from_biomodels(ID)
    assert len(biomodels.urls) == 4
    assert (tmp_path / f"{SHA}.xml").read_bytes() == SBML


def test_a_file_that_is_not_the_published_one_is_refused(biomodels):
    biomodels.meta = _meta(sha256="0" * 64)
    with pytest.raises(ModelError, match="not the"):
        bngsim.Model.from_biomodels(ID)


def test_a_file_that_is_not_sbml_is_refused(biomodels):
    page = b"<html>Request blocked</html>"
    biomodels.data = page
    biomodels.meta = _meta(sha256=hashlib.sha256(page).hexdigest())
    with pytest.raises(ModelError, match="is not SBML"):
        bngsim.Model.from_biomodels(ID)


def test_a_web_page_in_place_of_the_api_is_a_failed_fetch(biomodels):
    biomodels.meta = b"<html>Request blocked</html>"
    with pytest.raises(ModelError, match="could not fetch.*not its API's JSON"):
        bngsim.Model.from_biomodels(ID)
    with pytest.raises(ValueError, match="not a version number"):
        bngsim.Model.from_biomodels(ID, version=0)


def test_no_network_and_no_copy_says_so_and_names_the_cache(biomodels, tmp_path):
    biomodels.offline = True
    with pytest.raises(ModelError, match="could not fetch") as exc:
        bngsim.Model.from_biomodels(ID)
    assert str(tmp_path) in str(exc.value)
    assert "Model.from_sbml" in str(exc.value)


def test_a_missing_entry_and_a_malformed_id(biomodels):
    with pytest.raises(ModelError, match="no entry 'BIOMD9999999999'"):
        bngsim.Model.from_biomodels("BIOMD9999999999")
    with pytest.raises(ValueError, match="not a BioModels id"):
        bngsim.Model.from_biomodels("biomd10")


def test_load_says_to_call_from_biomodels():
    with pytest.raises(ModelError, match=r"Model.from_biomodels\('BIOMD0000000010'\)"):
        bngsim.Model.load("BIOMD0000000010")


def test_the_request_names_bngsim_and_nothing_about_the_user():
    agent = _biomodels._user_agent()
    assert agent == f"bngsim/{bngsim.__version__} (+https://github.com/lanl/bngsim)"
