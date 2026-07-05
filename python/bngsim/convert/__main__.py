"""Enable ``python -m bngsim.convert`` as an alias for the sbml2net CLI."""

from __future__ import annotations

from bngsim.convert._cli import main

if __name__ == "__main__":
    raise SystemExit(main())
