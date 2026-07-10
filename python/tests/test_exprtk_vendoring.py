"""Fast vendoring guardrails for the checked-in ExprTk header.

These checks are intentionally file-based rather than behavioral: they make
manual edits to `third_party/exprtk/exprtk.hpp` or stale vendoring metadata
fail fast before they turn into harder-to-debug parser/runtime regressions.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

BNGSIM_ROOT = Path(__file__).resolve().parents[2]
EXPRTK_DIR = BNGSIM_ROOT / "third_party" / "exprtk"
EXPRTK_HEADER = EXPRTK_DIR / "exprtk.hpp"
EXPRTK_VENDOR_METADATA = EXPRTK_DIR / "VENDOR.json"
EXPECTED_REQUIRED_TOKENS = [
    "static const std::string reserved_words[]",
    "static const std::string reserved_symbols[]",
    "#ifndef exprtk_disable_caseinsensitivity",
    "#ifndef exprtk_disable_string_capabilities",
    "#ifndef exprtk_disable_rtl_io_file",
    "#ifndef exprtk_disable_rtl_vecops",
    "class ifunction : public function_traits",
    "allow_zero_parameters()",
    "set_max_stack_depth",
]
EXPECTED_WRAPPER_ALIASES = ["ln", "rint", "sign", "mratio", "time"]


def _extract_string_array(header_text: str, array_name: str) -> list[str]:
    pattern = re.compile(
        rf"static const std::string {re.escape(array_name)}\[\]\s*=\s*\{{(?P<body>.*?)\n\s*\}};",
        re.DOTALL,
    )
    match = pattern.search(header_text)
    assert match, f"Could not extract ExprTk array '{array_name}' from exprtk.hpp"
    return re.findall(r'"([^"]+)"', match.group("body"))


def test_exprtk_vendor_metadata_matches_header():
    metadata = json.loads(EXPRTK_VENDOR_METADATA.read_text())
    header_bytes = EXPRTK_HEADER.read_bytes()
    header_text = header_bytes.decode("utf-8")

    assert (
        metadata["source"]["authoritative_remote"] == "https://github.com/ArashPartow/exprtk.git"
    )
    assert metadata["source"]["authoritative_branch"] == "master"
    assert metadata["source"]["official_project_homepage"] == (
        "https://www.partow.net/programming/exprtk/index.html"
    )
    assert (
        metadata["source"]["official_project_download"]
        == "https://www.partow.net/downloads/exprtk.zip"
    )

    assert metadata["files"]["header"]["path"] == "bngsim/third_party/exprtk/exprtk.hpp"
    assert metadata["files"]["header"]["sha256"] == hashlib.sha256(header_bytes).hexdigest()
    assert metadata["files"]["header"]["bytes"] == len(header_bytes)

    assert metadata["guardrails"]["reserved_words"] == _extract_string_array(
        header_text, "reserved_words"
    )
    assert metadata["guardrails"]["reserved_symbols"] == _extract_string_array(
        header_text, "reserved_symbols"
    )
    assert metadata["guardrails"]["required_header_tokens"] == EXPECTED_REQUIRED_TOKENS
    assert metadata["guardrails"]["bngsim_wrapper_aliases"] == EXPECTED_WRAPPER_ALIASES
    assert metadata["local_carries"] == []

    missing = [token for token in EXPECTED_REQUIRED_TOKENS if token not in header_text]
    assert missing == []
