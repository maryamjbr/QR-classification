"""Educational full-QR factorization and column-update routines.

The Givens convention follows Hammarling and Lucas (2008): ``givens(a, b)``
returns ``c, s`` for ``G = [[c, s], [-s, c]]`` such that
``G.T @ [a, b] == [d, 0]`` (up to floating-point rounding).  Consequently,
the update applies ``G.T`` to rows of ``R`` and ``G`` to columns of ``Q``.
"""

from __future__ import annotations

import numpy as np


def householder_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the full QR factorization ``A = Q @ R``.

    Parameters
    ----------
    A:
        Real two-dimensional array with shape ``(m, n)``.

    Returns
    -------
    Q, R:
        ``Q`` has shape ``(m, m)`` and is orthogonal. ``R`` has shape
        ``(m, n)`` and is upper trapezoidal.

    Notes
    -----
    The implementation forms the full ``Q`` because the subsequent column
    update needs coordinates in its orthogonal complement. Householder
    vectors with zero norm are skipped safely.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional array")

    dtype = np.result_type(A.dtype, np.float64)
    R = np.array(A, dtype=dtype, copy=True)
    m, n = R.shape
    Q = np.eye(m, dtype=dtype)

    for k in range(min(m, n)):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            continue

        sign = 1.0 if x[0] >= 0 else -1.0
        v = x.copy()
        v[0] += sign * norm_x
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            continue
        v /= norm_v

        trailing_R = R[k:, k:]
        R[k:, k:] = trailing_R - 2.0 * np.outer(v, v @ trailing_R)

        trailing_Q = Q[:, k:]
        Q[:, k:] = trailing_Q - 2.0 * np.outer(trailing_Q @ v, v)

    return Q, R


def givens(a: float, b: float) -> tuple[float, float]:
    """Return a stable Hammarling--Lucas Givens rotation.

    For ``G = np.array([[c, s], [-s, c]])``, the returned values satisfy
    ``G.T @ np.array([a, b]) == np.array([d, 0])`` up to rounding. The
    scalar ``d`` is allowed to have either sign.
    """
    a = float(a)
    b = float(b)
    if b == 0.0:
        return 1.0, 0.0

    if abs(b) >= abs(a):
        tau = -a / b
        s = 1.0 / np.sqrt(1.0 + tau * tau)
        c = s * tau
    else:
        tau = -b / a
        c = 1.0 / np.sqrt(1.0 + tau * tau)
        s = c * tau
    return c, s


def _append_column_in_place(
    Q: np.ndarray, R: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Append one column to copied factors; internal in-place helper."""
    m = Q.shape[0]
    current_n = R.shape[1]
    projected = Q.T @ x
    R_aug = np.column_stack((R, projected))

    # If current_n >= m, the appended column has no subdiagonal entries:
    # an m-by-(current_n + 1) upper-trapezoidal matrix may be dense there.
    if current_n >= m:
        return Q, R_aug

    pivot = current_n
    for row in range(pivot + 1, m):
        b = R_aug[row, pivot]
        if b == 0.0:
            continue
        c, s = givens(R_aug[pivot, pivot], b)
        G = np.array(((c, s), (-s, c)), dtype=R_aug.dtype)
        rows = [pivot, row]

        # givens() follows the paper's G.T convention.
        R_aug[rows, pivot:] = G.T @ R_aug[rows, pivot:]
        Q[:, rows] = Q[:, rows] @ G
        R_aug[row, pivot] = 0.0

    return Q, R_aug


def _validate_update_inputs(
    Q: np.ndarray, R: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = np.asarray(Q)
    R = np.asarray(R)
    x = np.asarray(x)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square two-dimensional array")
    if R.ndim != 2 or R.shape[0] != Q.shape[0]:
        raise ValueError("R must be two-dimensional with the same row count as Q")
    if x.ndim != 1 or x.shape[0] != Q.shape[0]:
        raise ValueError("x must be a vector with one entry per row of Q")

    dtype = np.result_type(Q.dtype, R.dtype, x.dtype, np.float64)
    return (
        np.array(Q, dtype=dtype, copy=True),
        np.array(R, dtype=dtype, copy=True),
        np.array(x, dtype=dtype, copy=False),
    )


def update_qr_with_new_column(
    Q: np.ndarray, R: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Update a full QR factorization after appending one column.

    Given ``A = Q @ R``, where ``Q`` is ``(m, m)`` and ``R`` is the
    ``(m, n)`` upper-trapezoidal factor, return full factors for ``[A, x]``.
    Inputs are not modified.
    """
    Q_new, R_new, x = _validate_update_inputs(Q, R, x)
    return _append_column_in_place(Q_new, R_new, x)


def update_qr_with_givens(
    Q: np.ndarray, R: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Update a full QR factorization after appending a block of columns.

    ``X`` must have shape ``(m, p)``. Its columns are appended sequentially.
    The input arrays are not modified.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if X.shape[0] != np.asarray(Q).shape[0]:
        raise ValueError("X must have the same row count as Q")

    dtype = np.result_type(np.asarray(Q).dtype, np.asarray(R).dtype, X.dtype)
    Q_new, R_new, _ = _validate_update_inputs(
        Q, R, np.zeros(np.asarray(Q).shape[0], dtype=dtype)
    )
    X = np.asarray(X, dtype=Q_new.dtype)
    for column in range(X.shape[1]):
        Q_new, R_new = _append_column_in_place(
            Q_new, R_new, X[:, column]
        )
    return Q_new, R_new


def qr_diagnostics(
    A: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> dict[str, float]:
    """Return absolute reconstruction, orthogonality, and triangularity errors."""
    A = np.asarray(A)
    Q = np.asarray(Q)
    R = np.asarray(R)
    return {
        "reconstruction": float(np.linalg.norm(A - Q @ R)),
        "orthogonality": float(
            np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1], dtype=Q.dtype))
        ),
        "triangularity": float(np.linalg.norm(R - np.triu(R))),
    }
