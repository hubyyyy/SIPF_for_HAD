import unittest

import jax.numpy as jnp
import numpy as np

from cancer_sipf.spectral import (
    build_spectral_grid,
    coefficients_to_grid,
    enforce_real_spectrum,
    galerkin_product,
)


class SpectralTests(unittest.TestCase):
    def test_constant_galerkin_product(self):
        grid = build_spectral_grid(6, 1.0, 2)
        left = jnp.zeros((6, 6, 6), dtype=jnp.complex64).at[0, 0, 0].set(2.0)
        right = jnp.zeros((6, 6, 6), dtype=jnp.complex64).at[0, 0, 0].set(3.0)
        product = np.asarray(galerkin_product(left, right, grid))
        expected = np.zeros_like(product)
        expected[0, 0, 0] = 6.0
        np.testing.assert_allclose(product, expected, atol=1.0e-6)

    def test_nonconstant_galerkin_product_matches_valid_mode_sum(self):
        grid = build_spectral_grid(6, 1.0, 2)
        left = np.zeros((6, 6, 6), dtype=np.complex64)
        right = np.zeros_like(left)
        left[1, 0, 0] = left[-1, 0, 0] = 2.0
        right[1, 0, 0] = right[-1, 0, 0] = 3.0
        product = np.asarray(
            galerkin_product(jnp.asarray(left), jnp.asarray(right), grid)
        )
        expected = np.zeros_like(product)
        expected[0, 0, 0] = 12.0
        expected[2, 0, 0] = 6.0
        expected[-2, 0, 0] = 6.0
        np.testing.assert_allclose(product, expected, atol=2.0e-6)

    def test_real_spectrum_is_real_off_grid_compatible(self):
        size = 8
        raw = jnp.arange(size**3, dtype=jnp.float32).reshape((size,) * 3)
        coefficients = enforce_real_spectrum(raw + 0.25j * raw)
        values = np.asarray(coefficients_to_grid(coefficients))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values.dtype.kind, "f")
        coefficients = np.asarray(coefficients)
        self.assertTrue(np.all(coefficients[size // 2, :, :] == 0.0))
        self.assertTrue(np.all(coefficients[:, size // 2, :] == 0.0))
        self.assertTrue(np.all(coefficients[:, :, size // 2] == 0.0))


if __name__ == "__main__":
    unittest.main()
