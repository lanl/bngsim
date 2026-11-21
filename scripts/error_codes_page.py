"""Write the error-code reference page from bngsim._error_codes (issue #1702).

``docs/reference/error-codes.md`` is generated, and ``test_error_codes.py``
fails while it differs from what :func:`bngsim._error_codes.reference_page`
gives. After a change to ``bngsim/_error_codes.py``, run::

    python scripts/error_codes_page.py
"""

from __future__ import annotations

from pathlib import Path

from bngsim import _error_codes

PAGE = Path(__file__).resolve().parents[1] / _error_codes.PAGE


def main() -> int:
    text = _error_codes.reference_page()
    if PAGE.exists() and PAGE.read_text(encoding="utf-8") == text:
        print(f"{PAGE}: current")
    else:
        PAGE.write_text(text, encoding="utf-8")
        print(f"{PAGE}: written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
