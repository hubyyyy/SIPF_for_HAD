"""Initial particles and fields from the numerical experiments in the paper."""

from __future__ import annotations

import math
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from .config import SimulationConfig
from .spectral import SpectralGrid


def single_cluster_mass(epsilon: float, radius: float) -> float:
    """Return the integral of exp(-r^2/epsilon) over a truncated 3D ball."""
    scaled_radius = radius / math.sqrt(epsilon)
    radial_integral = (
        math.sqrt(math.pi) * epsilon**1.5 * math.erf(scaled_radius) / 4.0
        - epsilon * radius * math.exp(-(scaled_radius**2)) / 2.0
    )
    return 4.0 * math.pi * radial_integral


def total_cell_mass(config: SimulationConfig) -> float:
    return len(config.centers) * single_cluster_mass(
        config.epsilon, config.support_radius
    )


def sample_initial_particles(config: SimulationConfig):
    """Sample the normalized truncated-Gaussian mixture in physical coordinates."""
    rng = np.random.default_rng(config.seed)
    table_size = 16_385
    radii = np.linspace(0.0, config.support_radius, table_size)
    radial_density = radii**2 * np.exp(-(radii**2) / config.epsilon)
    cdf = np.zeros_like(radii)
    increments = 0.5 * (radial_density[1:] + radial_density[:-1]) * np.diff(radii)
    cdf[1:] = np.cumsum(increments)
    cdf /= cdf[-1]

    sampled_radii = np.interp(rng.random(config.particles), cdf, radii)
    directions = rng.normal(size=(config.particles, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    cluster_ids = rng.integers(0, len(config.centers), size=config.particles)
    centers = np.asarray(config.centers, dtype=np.float64)
    physical_positions = centers[cluster_ids] + sampled_radii[:, None] * directions

    # Internally the FFT uses y=x+L/2 in [0,L); outputs are shifted back to x.
    fft_positions = np.mod(
        physical_positions + 0.5 * config.box_length, config.box_length
    )
    dtype = np.float64 if config.enable_x64 else np.float32
    return jnp.asarray(fft_positions, dtype=dtype)


def _initial_density_grid(
    config: SimulationConfig, grid_size: int
) -> np.ndarray:
    coordinates = (
        np.arange(grid_size, dtype=np.float64) * config.box_length / grid_size
        - 0.5 * config.box_length
    )
    x = coordinates[:, None, None]
    y = coordinates[None, :, None]
    z = coordinates[None, None, :]
    density = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    radius_squared = config.support_radius**2
    for center in config.centers:
        distance_squared = (
            (x - center[0]) ** 2
            + (y - center[1]) ** 2
            + (z - center[2]) ** 2
        )
        density += np.exp(-distance_squared / config.epsilon) * (
            distance_squared <= radius_squared
        )
    return density


def _extract_modes(fine_coefficients: np.ndarray, modes: np.ndarray) -> np.ndarray:
    fine_size = fine_coefficients.shape[0]
    indices = np.mod(modes, fine_size)
    return fine_coefficients[np.ix_(indices, indices, indices)]


def initial_field_coefficients(
    config: SimulationConfig, grid: SpectralGrid
) -> Tuple[object, object]:
    """Project f0=1-rho0/2 and m0=rho0/2 onto the retained modes."""
    fine_size = config.initial_grid_factor * config.modes
    density = _initial_density_grid(config, fine_size)
    f_values = 1.0 - 0.5 * density
    m_values = 0.5 * density
    normalization = fine_size**3
    f_fine = np.fft.fftn(f_values) / normalization
    m_fine = np.fft.fftn(m_values) / normalization
    modes = np.asarray(grid.modes)
    complex_dtype = np.complex128 if config.enable_x64 else np.complex64
    f_hat = _extract_modes(f_fine, modes).astype(complex_dtype)
    m_hat = _extract_modes(m_fine, modes).astype(complex_dtype)
    mean_half_density = 0.5 * total_cell_mass(config) / config.box_length**3
    f_hat[0, 0, 0] = 1.0 - mean_half_density
    m_hat[0, 0, 0] = mean_half_density
    return jnp.asarray(f_hat), jnp.asarray(m_hat)
