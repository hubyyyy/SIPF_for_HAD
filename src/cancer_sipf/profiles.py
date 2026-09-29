"""Radial profiles for cell-centered FDM grids and FFT-based SIPF grids."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional, Union

import numpy as np


class RadialProfile(NamedTuple):
    radius: np.ndarray
    mean: np.ndarray
    count: np.ndarray
    edges: np.ndarray


def radial_average_grid(
    values: np.ndarray,
    box_length: float,
    bins: Union[int, np.ndarray],
    coordinate_convention: str,
    max_radius: Optional[float] = None,
) -> RadialProfile:
    """Average a cubic 3D field over spherical radial bins."""
    values = np.asarray(values)
    if values.ndim != 3 or len(set(values.shape)) != 1:
        raise ValueError("values must be a cubic three-dimensional array")
    size = values.shape[0]
    if max_radius is None:
        max_radius = 0.5 * box_length
    if max_radius <= 0.0:
        raise ValueError("max_radius must be positive")
    if coordinate_convention == "cell_centered":
        coordinates = (
            (np.arange(size, dtype=np.float64) + 0.5) * box_length / size
            - 0.5 * box_length
        )
    elif coordinate_convention == "fft":
        coordinates = (
            np.arange(size, dtype=np.float64) * box_length / size
            - 0.5 * box_length
        )
    else:
        raise ValueError("coordinate_convention must be 'cell_centered' or 'fft'")

    if np.isscalar(bins):
        bin_count = int(bins)
        if bin_count < 1:
            raise ValueError("bins must be positive")
        edges = np.linspace(0.0, max_radius, bin_count + 1)
    else:
        edges = np.asarray(bins, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("bin edges must be a strictly increasing 1D array")
        bin_count = edges.size - 1

    radius = np.sqrt(
        coordinates[:, None, None] ** 2
        + coordinates[None, :, None] ** 2
        + coordinates[None, None, :] ** 2
    )
    flat_radius = radius.ravel()
    flat_values = np.real(values).ravel()
    indices = np.searchsorted(edges, flat_radius, side="right") - 1
    inside = (indices >= 0) & (indices < bin_count)
    indices = indices[inside]
    counts = np.bincount(indices, minlength=bin_count).astype(np.int64)
    value_sums = np.bincount(
        indices, weights=flat_values[inside], minlength=bin_count
    )
    radius_sums = np.bincount(
        indices, weights=flat_radius[inside], minlength=bin_count
    )
    means = np.full(bin_count, np.nan, dtype=np.float64)
    representative_radius = 0.5 * (edges[:-1] + edges[1:])
    nonempty = counts > 0
    means[nonempty] = value_sums[nonempty] / counts[nonempty]
    representative_radius[nonempty] = radius_sums[nonempty] / counts[nonempty]
    return RadialProfile(
        radius=representative_radius,
        mean=means,
        count=counts,
        edges=edges,
    )


def particle_radial_density(
    positions: np.ndarray,
    total_mass: float,
    edges: np.ndarray,
) -> RadialProfile:
    """Convert centered particle positions into shell-averaged density."""
    positions = np.asarray(positions)
    edges = np.asarray(edges, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (P,3)")
    radii = np.linalg.norm(positions, axis=1)
    counts, _ = np.histogram(radii, bins=edges)
    shell_volumes = 4.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    density = counts * (total_mass / positions.shape[0]) / shell_volumes
    return RadialProfile(
        radius=0.5 * (edges[:-1] + edges[1:]),
        mean=density,
        count=counts,
        edges=edges,
    )


def relative_l2_against_reference(
    profile: RadialProfile,
    reference_radius: np.ndarray,
    reference_values: np.ndarray,
) -> float:
    """Use the unweighted radial-bin relative L2 error stated in the paper."""
    reference = np.interp(
        profile.radius, np.asarray(reference_radius), np.asarray(reference_values)
    )
    valid = np.isfinite(profile.mean) & np.isfinite(reference)
    if not np.any(valid):
        raise ValueError("radial profile has no bins that can be compared")
    numerator = np.linalg.norm(profile.mean[valid] - reference[valid])
    denominator = np.linalg.norm(reference[valid])
    return float(numerator / max(denominator, 1.0e-15))


def save_profile(path: Union[str, Path], profile: RadialProfile, **extra) -> Path:
    path = Path(path)
    np.savez_compressed(
        path,
        radius=profile.radius,
        mean=profile.mean,
        count=profile.count,
        edges=profile.edges,
        **extra,
    )
    return path


def radialize_fdm_snapshot(
    snapshot_path: Union[str, Path],
    output_path: Union[str, Path],
    bins: Optional[int] = None,
    max_radius: Optional[float] = None,
) -> Path:
    """Create radial rho, f, and m profiles from one FDM snapshot."""
    with np.load(snapshot_path) as data:
        box_length = float(data["box_length"])
        rho = data["rho"]
        f = data["f"]
        m = data["m"]
        time = float(data["time"])
    if bins is None:
        bins = max(4, rho.shape[0] // 2)
    rho_profile = radial_average_grid(
        rho, box_length, bins, "cell_centered", max_radius
    )
    f_profile = radial_average_grid(
        f, box_length, rho_profile.edges, "cell_centered", max_radius
    )
    m_profile = radial_average_grid(
        m, box_length, rho_profile.edges, "cell_centered", max_radius
    )
    output_path = Path(output_path)
    np.savez_compressed(
        output_path,
        time=np.asarray(time),
        edges=rho_profile.edges,
        rho_radius=rho_profile.radius,
        rho_mean=rho_profile.mean,
        rho_count=rho_profile.count,
        f_radius=f_profile.radius,
        f_mean=f_profile.mean,
        f_count=f_profile.count,
        m_radius=m_profile.radius,
        m_mean=m_profile.mean,
        m_count=m_profile.count,
    )
    return output_path


def radialize_sipf_snapshot(
    snapshot_path: Union[str, Path],
    output_path: Union[str, Path],
    total_mass: Optional[float] = None,
    bins: Optional[int] = None,
    max_radius: Optional[float] = None,
) -> Path:
    """Create radial profiles while respecting SIPF's mixed grid conventions."""
    with np.load(snapshot_path) as data:
        box_length = float(data["box_length"])
        positions = data["positions"]
        rho_grid = data["rho_grid"]
        f_grid = data["f_grid"]
        m_grid = data["m_grid"]
        time = float(data["time"])
    if bins is None:
        bins = max(4, m_grid.shape[0] // 2)
    m_profile = radial_average_grid(
        m_grid, box_length, bins, "fft", max_radius
    )
    f_profile = radial_average_grid(
        f_grid, box_length, m_profile.edges, "fft", max_radius
    )
    rho_grid_profile = radial_average_grid(
        rho_grid, box_length, m_profile.edges, "cell_centered", max_radius
    )
    output = {
        "time": np.asarray(time),
        "edges": m_profile.edges,
        "rho_grid_radius": rho_grid_profile.radius,
        "rho_grid_mean": rho_grid_profile.mean,
        "rho_grid_count": rho_grid_profile.count,
        "f_radius": f_profile.radius,
        "f_mean": f_profile.mean,
        "f_count": f_profile.count,
        "m_radius": m_profile.radius,
        "m_mean": m_profile.mean,
        "m_count": m_profile.count,
    }
    if total_mass is not None:
        particle_profile = particle_radial_density(
            positions, total_mass, m_profile.edges
        )
        output.update(
            {
                "rho_particle_radius": particle_profile.radius,
                "rho_particle_mean": particle_profile.mean,
                "rho_particle_count": particle_profile.count,
            }
        )
    output_path = Path(output_path)
    np.savez_compressed(output_path, **output)
    return output_path
