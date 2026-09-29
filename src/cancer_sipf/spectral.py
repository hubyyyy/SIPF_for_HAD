"""Fourier operations used by the particle-field solver."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class SpectralGrid(NamedTuple):
    modes: object
    wave_numbers: object
    wave_number_squared: object
    pad_i: object
    pad_j: object
    pad_k: object
    size: int
    padded_size: int
    box_length: float


def build_spectral_grid(
    size: int, box_length: float, dealias_factor: int
) -> SpectralGrid:
    """Build standard FFT-ordered modes and indices for exact padded products."""
    modes = jnp.rint(jnp.fft.fftfreq(size) * size).astype(jnp.int32)
    wave_numbers = (2.0 * jnp.pi / box_length) * modes
    kx = wave_numbers[:, None, None]
    ky = wave_numbers[None, :, None]
    kz = wave_numbers[None, None, :]
    wave_number_squared = kx * kx + ky * ky + kz * kz

    padded_size = dealias_factor * size
    padded_indices = jnp.mod(modes, padded_size)
    pad_i, pad_j, pad_k = jnp.meshgrid(
        padded_indices, padded_indices, padded_indices, indexing="ij"
    )
    return SpectralGrid(
        modes=modes,
        wave_numbers=wave_numbers,
        wave_number_squared=wave_number_squared,
        pad_i=pad_i,
        pad_j=pad_j,
        pad_k=pad_k,
        size=size,
        padded_size=padded_size,
        box_length=box_length,
    )


def coefficients_to_grid(coefficients):
    """Evaluate a Fourier series on its native uniform grid."""
    size = coefficients.shape[0]
    return jnp.real(jnp.fft.ifftn(coefficients) * (size**3))


def enforce_real_spectrum(coefficients):
    """Enforce conjugate symmetry for a real continuous Fourier series."""
    size = coefficients.shape[0]
    negative_indices = jnp.mod(-jnp.arange(size), size)
    neg_i, neg_j, neg_k = jnp.meshgrid(
        negative_indices, negative_indices, negative_indices, indexing="ij"
    )
    symmetric = 0.5 * (
        coefficients + jnp.conj(coefficients[neg_i, neg_j, neg_k])
    )

    # An even FFT has only one of the two continuous Nyquist modes. Dropping
    # these planes keeps off-grid particle evaluations genuinely real.
    indices = jnp.arange(size)
    nyquist = size // 2
    nyquist_plane = (
        (indices[:, None, None] == nyquist)
        | (indices[None, :, None] == nyquist)
        | (indices[None, None, :] == nyquist)
    )
    return jnp.where(nyquist_plane, jnp.zeros_like(symmetric), symmetric)


def _pack_padded(coefficients, grid: SpectralGrid):
    padded = jnp.zeros(
        (grid.padded_size, grid.padded_size, grid.padded_size),
        dtype=coefficients.dtype,
    )
    return padded.at[grid.pad_i, grid.pad_j, grid.pad_k].set(coefficients)


def galerkin_product(left, right, grid: SpectralGrid):
    """Compute the valid-mode Fourier convolution without circular aliasing."""
    left_padded = _pack_padded(left, grid)
    right_padded = _pack_padded(right, grid)
    scale = grid.padded_size**3
    left_values = jnp.fft.ifftn(left_padded) * scale
    right_values = jnp.fft.ifftn(right_padded) * scale
    product_padded = jnp.fft.fftn(left_values * right_values) / scale
    product = product_padded[grid.pad_i, grid.pad_j, grid.pad_k]
    return enforce_real_spectrum(product)
