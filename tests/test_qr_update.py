import unittest

import numpy as np

from qr_update import (
    givens,
    householder_qr,
    qr_diagnostics,
    update_qr_with_givens,
    update_qr_with_new_column,
)


class GivensTests(unittest.TestCase):
    def test_documented_rotation_convention(self):
        c, s = givens(4.0, 3.0)
        G = np.array(((c, s), (-s, c)))
        np.testing.assert_allclose(G.T @ [4.0, 3.0], [5.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(G @ G.T, np.eye(2), atol=1e-15)

    def test_zero_and_small_edge_cases(self):
        cases = (
            (0.0, 0.0),
            (4.0, 0.0),
            (0.0, 3.0),
            (1e-300, -2e-300),
            (-1e-300, 2e-300),
            (1e300, -2e300),
        )
        for a, b in cases:
            with self.subTest(a=a, b=b):
                c, s = givens(a, b)
                G = np.array(((c, s), (-s, c)))
                rotated = G.T @ np.array([a, b])
                scale = max(abs(a), abs(b), np.finfo(float).tiny)
                self.assertLessEqual(abs(rotated[1]) / scale, 5e-16)
                np.testing.assert_allclose(G @ G.T, np.eye(2), atol=1e-15)


class HouseholderTests(unittest.TestCase):
    def test_zero_norm_vectors_are_safe(self):
        A = np.zeros((6, 3))
        Q, R = householder_qr(A)
        np.testing.assert_array_equal(Q, np.eye(6))
        np.testing.assert_array_equal(R, A)

    def test_zero_trailing_column_is_safe(self):
        A = np.array(
            [
                [2.0, 0.0, 1.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0],
            ]
        )
        Q, R = householder_qr(A)
        errors = qr_diagnostics(A, Q, R)
        self.assertLess(errors["reconstruction"], 1e-12)
        self.assertLess(errors["orthogonality"], 1e-12)
        self.assertLess(errors["triangularity"], 1e-12)


class IncrementalUpdateTests(unittest.TestCase):
    def assert_valid_update(self, m, n, added, seed):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(m, n))
        X = rng.normal(size=(m, added))
        A_new = np.column_stack((A, X))

        Q, R = np.linalg.qr(A, mode="complete")
        Q_new, R_new = update_qr_with_givens(Q, R, X)
        errors = qr_diagnostics(A_new, Q_new, R_new)

        scale = max(np.linalg.norm(A_new), 1.0)
        self.assertLess(errors["reconstruction"] / scale, 2e-13)
        self.assertLess(errors["orthogonality"], 2e-13)
        self.assertLess(errors["triangularity"] / scale, 2e-13)

        # QR factors are sign-indeterminate, so compare projectors rather than
        # Q and R element by element.
        Q_reference, _ = np.linalg.qr(A_new, mode="reduced")
        active_columns = min(m, n + added)
        projector_update = Q_new[:, :active_columns] @ Q_new[:, :active_columns].T
        projector_reference = Q_reference @ Q_reference.T
        np.testing.assert_allclose(
            projector_update, projector_reference, atol=2e-12, rtol=2e-12
        )

    def test_random_tall_and_square_updates(self):
        cases = (
            (8, 3, 2, 1),
            (10, 1, 3, 2),
            (7, 5, 1, 3),
            (6, 6, 2, 4),
        )
        for case in cases:
            with self.subTest(case=case):
                self.assert_valid_update(*case)

    def test_already_zero_tail(self):
        rng = np.random.default_rng(5)
        A = rng.normal(size=(8, 3))
        Q, R = np.linalg.qr(A, mode="complete")
        coordinates = np.zeros(8)
        coordinates[:4] = rng.normal(size=4)
        x = Q @ coordinates
        Q_new, R_new = update_qr_with_new_column(Q, R, x)
        errors = qr_diagnostics(np.column_stack((A, x)), Q_new, R_new)
        self.assertLess(errors["reconstruction"], 1e-12)
        self.assertLess(errors["triangularity"], 1e-12)

    def test_very_small_new_column(self):
        rng = np.random.default_rng(7)
        A = rng.normal(size=(8, 3))
        x = rng.normal(size=8) * 1e-200
        Q, R = np.linalg.qr(A, mode="complete")
        Q_new, R_new = update_qr_with_new_column(Q, R, x)
        errors = qr_diagnostics(np.column_stack((A, x)), Q_new, R_new)
        self.assertLess(errors["reconstruction"], 1e-12)
        self.assertLess(errors["orthogonality"], 1e-12)
        self.assertLess(errors["triangularity"], 1e-12)

    def test_inputs_are_not_modified(self):
        rng = np.random.default_rng(6)
        A = rng.normal(size=(9, 4))
        x = rng.normal(size=9)
        Q, R = np.linalg.qr(A, mode="complete")
        Q_before = Q.copy()
        R_before = R.copy()
        update_qr_with_new_column(Q, R, x)
        np.testing.assert_array_equal(Q, Q_before)
        np.testing.assert_array_equal(R, R_before)


if __name__ == "__main__":
    unittest.main()
