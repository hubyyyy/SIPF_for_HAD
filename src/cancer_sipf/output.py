"""Portable simulation output and lightweight visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import numpy as np

from .config import SimulationConfig
from .solver import SIPFSolver, SimulationState


def write_json(path: Path, values: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
        handle.write("\n")


def prepare_output_directory(config: SimulationConfig) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config.to_dict())
    return output_dir


def save_snapshot(
    output_dir: Path, solver: SIPFSolver, state: SimulationState
) -> Path:
    config = solver.config
    step = int(np.asarray(jax.device_get(state.step)))
    positions = np.asarray(
        jax.device_get(solver.centered_positions(state.positions))
    )
    f_hat = np.asarray(jax.device_get(state.f_hat))
    m_hat = np.asarray(jax.device_get(state.m_hat))
    f_grid = np.asarray(jax.device_get(solver.field_on_grid(state.f_hat)))
    m_grid = np.asarray(jax.device_get(solver.field_on_grid(state.m_hat)))

    half_box = 0.5 * config.box_length
    edges = np.linspace(-half_box, half_box, config.modes + 1)
    counts, _ = np.histogramdd(positions, bins=(edges, edges, edges))
    cell_volume = (config.box_length / config.modes) ** 3
    rho_grid = counts * (solver.mass / config.particles) / cell_volume

    path = output_dir / f"snapshot_{step:06d}.npz"
    np.savez_compressed(
        path,
        step=np.asarray(step, dtype=np.int64),
        time=np.asarray(step * config.dt),
        box_length=np.asarray(config.box_length),
        positions=positions,
        rho_grid=rho_grid,
        f_grid=f_grid,
        m_grid=m_grid,
        f_hat=f_hat,
        m_hat=m_hat,
    )
    return path


def plot_snapshot(
    snapshot_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
):
    """Plot central x-z slices and a particle projection from one NPZ snapshot."""
    import matplotlib.pyplot as plt

    snapshot_path = Path(snapshot_path)
    with np.load(snapshot_path) as data:
        positions = data["positions"]
        rho = data["rho_grid"]
        f_grid = data["f_grid"]
        m_grid = data["m_grid"]
        time = float(data["time"])
        box_length = float(data["box_length"])

    center = rho.shape[1] // 2
    half_box = 0.5 * box_length
    extent = (-half_box, half_box, -half_box, half_box)
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)
    axes[0].scatter(positions[:, 0], positions[:, 2], s=1, alpha=0.25)
    axes[0].set_title("Particles")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")
    axes[0].set_aspect("equal")

    for axis, values, title in zip(
        axes[1:],
        (rho[:, center, :], f_grid[:, center, :], m_grid[:, center, :]),
        ("Cell density", "ECM f", "MDE m"),
    ):
        image = axis.imshow(
            values.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("z")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"SIPF snapshot at t={time:g}")

    if output_path is None:
        output_path = snapshot_path.with_suffix(".png")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return Path(output_path)
