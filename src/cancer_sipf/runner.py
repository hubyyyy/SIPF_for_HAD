"""High-level simulation loop."""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Callable, Dict

import jax

from .config import SimulationConfig
from .output import prepare_output_directory, save_snapshot, write_json
from .solver import SIPFSolver


def _block_until_ready(value) -> None:
    value.block_until_ready()


def run_simulation(
    config: SimulationConfig, log: Callable[[str], None] = print
) -> Dict[str, object]:
    """Run one configured simulation and write portable NPZ snapshots."""
    start = time.perf_counter()
    config.validate()
    output_dir = prepare_output_directory(config)
    solver = SIPFSolver(config)
    state = solver.initial_state()
    devices = [str(device) for device in jax.devices()]

    log(
        f"Backend: {jax.default_backend()} | devices: {', '.join(devices)}"
    )
    log(
        f"SIPF: P={config.particles}, H={config.modes}, "
        f"dt={config.dt:g}, steps={config.steps}"
    )
    save_snapshot(output_dir, solver, state)

    first_block_seconds = None
    completed = 0
    while completed < config.steps:
        block_steps = min(config.save_every, config.steps - completed)
        block_start = time.perf_counter()
        state = solver.advance(state, block_steps)
        _block_until_ready(state.positions)
        block_seconds = time.perf_counter() - block_start
        if first_block_seconds is None:
            first_block_seconds = block_seconds
        completed += block_steps
        snapshot_path = save_snapshot(output_dir, solver, state)
        diagnostics = solver.diagnostics(state)
        log(
            f"t={diagnostics['time']:.4g} | step={completed}/{config.steps} | "
            f"block={block_seconds:.3f}s | "
            f"MDE mass error={diagnostics['m_integral_relative_error']:.2e} | "
            f"{snapshot_path.name}"
        )

    wall_seconds = time.perf_counter() - start
    diagnostics = solver.diagnostics(state)
    summary: Dict[str, object] = {
        "backend": jax.default_backend(),
        "devices": devices,
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "wall_seconds_including_compilation": wall_seconds,
        "first_block_seconds_including_compilation": first_block_seconds,
        "completed_steps": config.steps,
        "final_diagnostics": diagnostics,
    }
    write_json(Path(output_dir) / "summary.json", summary)
    log(f"Finished in {wall_seconds:.3f}s; output: {output_dir}")
    return summary
