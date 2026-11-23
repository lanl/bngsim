"""bngsim.NamedArray — RoadRunner-compatible labeled ndarray.

Mirrors the shape of :class:`roadrunner.NamedArray`: a 2-D
:class:`numpy.ndarray` subclass that carries column names and
supports column lookup by string. Used by
:meth:`bngsim.Result.as_roadrunner` to provide a drop-in replacement
for ``rr.simulate(...)`` output in PyBNF-style stochastic-fitting
workflows.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from bngsim._ndarray_pickle import unwrap_state, wrap_state

#: Marks a pickle state written by :meth:`NamedArray.__reduce__`. A stream
#: written before issue #629 carries numpy's own state and no names, and
#: :meth:`NamedArray.__setstate__` tells the two apart by this tag rather than
#: by the state's shape, which is numpy's to change.
_PICKLE_TAG = "bngsim.NamedArray"
_PICKLE_VERSION = 1


class NamedArray(np.ndarray):
    """A 2-D ndarray with named columns (RoadRunner-compatible).

    Constructed via :meth:`bngsim.Result.as_roadrunner`. End users
    rarely instantiate it directly; subclass primarily exists so that
    callers can identify the array shape (``arr.colnames``) and look
    up columns by name (``arr["[X]"]``, ``arr[:, "[X]"]``).

    Attributes
    ----------
    colnames : list[str]
        One name per column, in column order — the invariant
        :meth:`__new__` checks, and one that holds for the lifetime of
        every array of this class (issue #561). Names are *positional*,
        so they are carried exactly where this class can see the whole
        operation and knows what became of the columns: construction;
        indexing, which relabels whatever survives the key
        (``arr[:, 1:]``, ``arr[:, ["time", "[X]"]]``, ``arr[:, ::-1]``,
        a row slice); and a pickle round trip, so an array that crosses
        a process boundary arrives labeled (issue #629).

        Any other array numpy hands back — a product, a transpose, a
        reduction, ``arr.copy()``, :func:`copy.deepcopy`,
        ``np.copy(arr, subok=True)``, ``astype``, ``view`` — has
        ``colnames == []`` and raises :class:`KeyError` on a lookup by
        name, rather than answering with whichever column now sits at
        the name's old position. A copy is in that list although its
        columns are plainly the parent's: ``.copy()`` is one of numpy's
        own primitives (``np.sort(a, axis=1)`` is a copy followed by an
        in-place sort), so a copy that carried labels would hand them to
        operations that permute the columns underneath them. The comment
        on the method that is *not* overridden has the measurement.

        libroadrunner keeps the labels through pickle as this class
        does, and drops them on every derived array, copies and slices
        included. Where this class keeps more of them, it is additive —
        code written against RoadRunner cannot depend on a label being
        absent — and it keeps one only where the column is established
        to be the same column.

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
    >>> arr[:, 1:].colnames             # the columns the slice kept
    ['[X]', '[Y]']
    """

    # __slots__ omitted: ndarray subclasses must not declare __slots__
    # (numpy stores subclass attributes via __array_finalize__).

    def __new__(cls, data: NDArray[np.float64], colnames: list[str]) -> NamedArray:
        arr = np.asarray(data, dtype=np.float64).view(cls)
        if arr.ndim != 2:
            from bngsim._exceptions import UsageError

            raise UsageError(f"NamedArray must be 2-D, got shape {arr.shape}")
        if arr.shape[1] != len(colnames):
            from bngsim._exceptions import UsageError

            raise UsageError(
                f"colnames length {len(colnames)} does not match number of columns {arr.shape[1]}"
            )
        arr.colnames = list(colnames)
        return arr

    def __array_finalize__(self, obj: NDArray[Any] | None) -> None:
        # Deliberately does *not* inherit the parent's names (issue #561).
        # They label columns by position, and numpy hands this hook no
        # description of the transform that produced ``self``: a result of the
        # parent's shape is as likely to be ``np.sort(parent, axis=1)``, with
        # the columns permuted, as ``parent * 2``, with them intact. Inheriting
        # them is what let a column-dropping slice keep one name per column of
        # the *parent*, so that ``arr[:, 1:]["[X]"]`` returned the ``[Y]``
        # column and nothing raised. Names are attached only where the
        # surviving columns are known: __new__ and __getitem__.
        self.colnames: list[str] = []

    def __reduce__(self) -> tuple[Any, ...]:
        # numpy's reduction carries the array and none of a subclass's
        # attributes, so the names used to be lost on every round trip through a
        # process boundary: a fit collecting worker results got its tables back
        # unlabeled, and every lookup by name raised (issue #629). The wrapper
        # is shared with JacobianMatrix and documented in
        # bngsim/_ndarray_pickle.py, including why this is __reduce__ rather
        # than __reduce_ex__; the round trip is pinned at each protocol in
        # test_named_array_colnames.py rather than guessed at here.
        reduction = cast("tuple[Any, ...]", super().__reduce__())
        return wrap_state(reduction, _PICKLE_TAG, _PICKLE_VERSION, list(self.colnames))

    def __setstate__(self, state: Any) -> None:
        tagged, colnames, inner = unwrap_state(state, _PICKLE_TAG, _PICKLE_VERSION, "NamedArray")
        # `tagged` is False for a stream written before #629: numpy's own state,
        # carrying no names. The array still loads; it simply arrives unlabeled,
        # as it did then. The flag rather than a None payload is issue #646.
        self.colnames = list(colnames) if tagged else []
        super().__setstate__(inner)

    # ``copy()`` deliberately has NO override, though a copy plainly has the
    # parent's columns and carrying the names across looks sound. It is not:
    # ``.copy()`` is one of numpy's own primitives, not merely a method users
    # call. ``np.sort(a, axis=1)`` is implemented as ``a.copy(order="K")``
    # followed by an in-place sort, and ``np.partition`` the same way, so a
    # name-preserving ``copy()`` labels the row-wise minima with the column
    # names — ``np.sort(arr, axis=1)["time"]`` answering with the minima of
    # each row is the issue #561 defect through a different door. It was
    # measured, not guessed: the "sorted within rows" case in
    # test_named_array_colnames.py fails the moment ``copy()`` carries names,
    # and it is there to keep this from being re-attempted.
    #
    # ``copy.copy`` / ``copy.deepcopy`` are not reachable that way — numpy's
    # own routines never go through the copy module — but a class where
    # ``arr.copy()`` and ``copy.copy(arr)`` disagree about labels is harder to
    # explain than one where no copy carries them, which is also what
    # libroadrunner does. So all three stay unlabeled.

    def __getitem__(self, key: Any) -> NDArray[np.float64]:  # type: ignore[override]
        # Forms supported:
        #   arr["name"]         → 1-D column   (RR convention)
        #   arr[:, "name"]      → 1-D column
        #   arr[i, "name"]      → scalar
        #   arr[i:j, ["a","b"]] → 2-D NamedArray slice
        # Anything else falls through to ndarray.__getitem__, and the names of
        # the columns that survived the key are computed back onto the result.
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
        out = super().__getitem__(key)
        if isinstance(out, NamedArray):
            out.colnames = self._colnames_after(key, out)
        return out

    def _colnames_after(self, key: Any, out: NamedArray) -> list[str]:
        """Names for the columns of ``self[key]``; ``[]`` when not derivable.

        Whatever selects along axis 1 of a 2-D array selects the same entries
        of a 1-D array of column positions, so basic and advanced indexing are
        both answered by indexing ``np.arange(ncols)`` with the key's column
        part. A key that leaves no recognizable column axis — one whose result
        is not 2-D, or a broadcast advanced index whose axis 1 is not the
        column axis — yields no names rather than a guess.
        """
        if out.ndim != 2 or not self.colnames:
            return []
        col_key: Any = slice(None)  # a key that indexes rows only keeps every column
        if isinstance(key, tuple):
            if len(key) > 2:
                return []
            if len(key) == 2:
                col_key = key[1]
        try:
            idxs = np.arange(len(self.colnames))[col_key]
        except (IndexError, TypeError, ValueError):
            return []
        if not isinstance(idxs, np.ndarray) or idxs.ndim != 1 or idxs.size != out.shape[1]:
            return []
        return [self.colnames[int(i)] for i in idxs]

    def _col_index(self, name: str) -> int:
        try:
            return self.colnames.index(name)
        except ValueError:
            from bngsim._exceptions import NotFoundError

            raise NotFoundError(
                self._unknown_selector_message(name),
                code="output.not_found",
                details={"name": name},
            ) from None

    def _col_by_name(self, name: str) -> NDArray[np.float64]:
        return np.asarray(self)[:, self._col_index(name)]

    def _unknown_selector_message(self, name: str) -> str:
        # Match RoadRunner's selector-not-found text closely enough that
        # PyBNF code that catches RR errors keeps working.
        if not self.colnames:
            return (
                f"Invalid selection '{name}'. This array carries no column names — it was "
                "either built without any, or derived by an operation whose effect on the "
                "columns is not tracked (arithmetic, a transpose, a reduction, unpickling), "
                "which drops the names rather than leaving them pointing at columns that may "
                "have moved. Look the column up on the array "
                "Result.as_roadrunner() returned, or index that array by name "
                "(arr[:, ['time', '[X]']]) to carry the names over."
            )
        return f"Invalid selection '{name}'. Valid selections: {self.colnames}"

    def __repr__(self) -> str:
        body = np.array2string(np.asarray(self), separator=", ")
        return f"NamedArray(\n{body},\ncolnames={self.colnames})"
