"""bngsim._evaluators — array conventions for the public model evaluators.

:meth:`bngsim.Model.rhs`, :meth:`~bngsim.Model.jacobian`,
:meth:`~bngsim.Model.propensities` and :meth:`~bngsim.Model.stoichiometry_matrix`
(issue #523) are thin wrappers over the C++ evaluators the integrators already
run. What lives here is the part that is about NumPy rather than about the
model: the Jacobian array type that says which evaluator built it, the
0-based conversion of the 1-based stoichiometry entry list, and the assembly
of a scipy CSC matrix from the model's own sparsity pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from bngsim._bngsim_core import NetworkModel

#: ``JacobianMatrix.source`` when the closed-form terms built the matrix.
ANALYTICAL = "analytical"
#: ``JacobianMatrix.source`` when a one-sided difference quotient built it —
#: the same spelling :attr:`bngsim.SteadyStateResult.solver_jacobian_source`
#: uses for the same fallback.
FINITE_DIFFERENCE = "finite-difference"


class JacobianMatrix(np.ndarray):
    """A dense ``(n_species, n_species)`` Jacobian that knows how it was built.

    Returned by :meth:`bngsim.Model.jacobian` with ``sparse=False``. An ordinary
    float64 :class:`numpy.ndarray` in every respect — ``J @ v``,
    ``numpy.linalg.eigvals(J)``, slicing — with one extra attribute:

    Attributes
    ----------
    source : str
        ``"analytical"`` when every reaction's derivative came from the model's
        closed form, ``"finite-difference"`` when the matrix is the one-sided
        difference quotient of :meth:`bngsim.Model.rhs`. A model whose
        analytical Jacobian is incomplete never hands back the partial matrix
        under the first label; it differences instead and says so.

    ``J[i, j]`` is ``∂f_i/∂x_j`` — row ``i`` is the species whose derivative is
    being differentiated, column ``j`` the species it is differentiated with
    respect to, both in :attr:`bngsim.Model.species_names` order.
    """

    # __slots__ omitted: ndarray subclasses must not declare __slots__
    # (numpy stores subclass attributes via __array_finalize__).

    source: str

    def __new__(cls, data: NDArray[np.float64], source: str) -> JacobianMatrix:
        arr = np.asarray(data, dtype=np.float64).view(cls)
        arr.source = source
        return arr

    def __array_finalize__(self, obj: NDArray[Any] | None) -> None:
        if obj is None:
            return
        self.source = getattr(obj, "source", "")


StoichCoo = tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]


def stoichiometry_coo(core: NetworkModel) -> StoichCoo:
    """The net stoichiometry as 0-based COO triplets ``(rows, cols, vals)``.

    Converted once from the C++ ``StoichEntry`` list, which is 1-based and holds
    one entry per ``(species, reaction)`` pair with the *net* coefficient
    (products positive, reactants negative). Rows of ``$``-fixed boundary
    species are dropped: the RHS zeroes their derivative and the SSA step never
    updates them, so the matrix that satisfies ``rhs(y) == S @ rates(y)`` — and
    the one conservation-law detection row-reduces — has an empty row there.
    """
    raw = core.stoichiometry()
    rows = np.asarray(raw["species_index"], dtype=np.int64) - 1
    cols = np.asarray(raw["reaction_index"], dtype=np.int64) - 1
    vals = np.asarray(raw["coefficient"], dtype=np.float64)
    fixed = np.asarray(core.species_is_fixed, dtype=bool)
    if fixed.any():
        keep = ~fixed[rows]
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
    return rows, cols, vals


def scipy_sparse() -> Any:
    """``scipy.sparse``, or an ImportError that names what ``sparse=True`` needs."""
    try:
        import scipy.sparse
    except ImportError as e:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "sparse=True returns a scipy.sparse.csc_array; install scipy "
            "(`pip install scipy`) or call without sparse=True for a dense ndarray"
        ) from e
    return scipy.sparse


def csc_from_pattern(core: NetworkModel, vals: NDArray[np.float64], source: str) -> Any:
    """A ``csc_array`` over the model's structural Jacobian pattern.

    ``vals`` is the value array in the pattern's own data order — what
    ``fill_sparse_analytical_jacobian`` writes, or what :func:`gather_pattern`
    picked out of a dense matrix — so the analytical and finite-difference
    results share one structure and a consumer can allocate for it once.
    """
    pat = core.jacobian_sparsity
    n = int(pat["n"])
    M = scipy_sparse().csc_array(
        (np.asarray(vals, dtype=np.float64), pat["row_indices"], pat["col_ptrs"]), shape=(n, n)
    )
    M.source = source
    return M


def gather_pattern(core: NetworkModel, dense: NDArray[np.float64]) -> NDArray[np.float64]:
    """The entries of a dense ``(n, n)`` matrix at the model's pattern positions.

    Lossless for a matrix the RHS produced: an entry outside the structural
    pattern is a derivative of a species with respect to one it does not read,
    and a difference quotient there is exactly zero — the two RHS evaluations
    run the same arithmetic on that component.
    """
    pat = core.jacobian_sparsity
    col_ptrs = np.asarray(pat["col_ptrs"], dtype=np.int64)
    rows = np.asarray(pat["row_indices"], dtype=np.int64)
    cols = np.repeat(np.arange(int(pat["n"]), dtype=np.int64), np.diff(col_ptrs))
    return np.ascontiguousarray(dense[rows, cols], dtype=np.float64)
