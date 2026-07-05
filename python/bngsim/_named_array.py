"""bngsim.NamedArray — RoadRunner-compatible labeled ndarray.

Mirrors the shape of :class:`roadrunner.NamedArray`: a 2-D
:class:`numpy.ndarray` subclass that carries column names and
supports column lookup by string. Used by
:meth:`bngsim.Result.as_roadrunner` to provide a drop-in replacement
for ``rr.simulate(...)`` output in PyBNF-style stochastic-fitting
workflows.

See ``dev/plans/SBML_SSA_SUPPORT_PLAN.md`` Phase 4.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class NamedArray(np.ndarray):
    """A 2-D ndarray with named columns (RoadRunner-compatible).

    Constructed via :meth:`bngsim.Result.as_roadrunner`. End users
    rarely instantiate it directly; subclass primarily exists so that
    callers can identify the array shape (``arr.colnames``) and look
    up columns by name (``arr["[X]"]``, ``arr[:, "[X]"]``).

    Attributes
    ----------
    colnames : list[str]
        One entry per column, in column order. Always a fresh list —
        slicing / arithmetic that drops columns invalidates the
        original mapping, so the inherited list is left untouched on
        non-trivial transforms.

    Examples
    --------
    >>> result = sim.run(t_span=(0, 10), n_points=11)
    >>> arr = result.as_roadrunner()
    >>> arr.colnames
    ['time', '[X]', '[Y]']
    >>> arr["[X]"]                      # 1-D column view
    array([ ... ])
    >>> arr[:, "[X]"]                   # equivalent
    array([ ... ])
    """

    # __slots__ omitted: ndarray subclasses must not declare __slots__
    # (numpy stores subclass attributes via __array_finalize__).

    def __new__(cls, data: NDArray[np.float64], colnames: list[str]) -> NamedArray:
        arr = np.asarray(data, dtype=np.float64).view(cls)
        if arr.ndim != 2:
            raise ValueError(f"NamedArray must be 2-D, got shape {arr.shape}")
        if arr.shape[1] != len(colnames):
            raise ValueError(
                f"colnames length {len(colnames)} does not match number of columns {arr.shape[1]}"
            )
        arr.colnames = list(colnames)
        return arr

    def __array_finalize__(self, obj: NDArray[Any] | None) -> None:
        if obj is None:
            return
        self.colnames = list(getattr(obj, "colnames", []))

    def __getitem__(self, key: Any) -> NDArray[np.float64]:  # type: ignore[override]
        # Forms supported:
        #   arr["name"]         → 1-D column   (RR convention)
        #   arr[:, "name"]      → 1-D column
        #   arr[i, "name"]      → scalar
        #   arr[i:j, ["a","b"]] → 2-D NamedArray slice
        # Anything else falls through to ndarray.__getitem__.
        if isinstance(key, str):
            return self._col_by_name(key)
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            if isinstance(col_key, str):
                idx = self._col_index(col_key)
                return np.asarray(self).__getitem__((row_key, idx))
            if isinstance(col_key, list) and col_key and isinstance(col_key[0], str):
                idxs = [self._col_index(name) for name in col_key]
                sub = np.asarray(self).__getitem__((row_key, idxs))
                if sub.ndim == 2:
                    return NamedArray(sub, [self.colnames[i] for i in idxs])
                return sub
        return super().__getitem__(key)

    def _col_index(self, name: str) -> int:
        try:
            return self.colnames.index(name)
        except ValueError:
            raise KeyError(self._unknown_selector_message(name)) from None

    def _col_by_name(self, name: str) -> NDArray[np.float64]:
        return np.asarray(self)[:, self._col_index(name)]

    def _unknown_selector_message(self, name: str) -> str:
        # Match RoadRunner's selector-not-found text closely enough that
        # PyBNF code that catches RR errors keeps working.
        return f"Invalid selection '{name}'. Valid selections: {self.colnames}"

    def __repr__(self) -> str:
        body = np.array2string(np.asarray(self), separator=", ")
        return f"NamedArray(\n{body},\ncolnames={self.colnames})"
