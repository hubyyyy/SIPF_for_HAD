"""Conservative three-dimensional finite-difference comparison solver."""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


@dataclass(frozen=True)
class FDMConfig:
    """Configuration for the cell-centered 3D FDM experiment."""

    d1: float = 0.001
    gamma: float = 0.005
    eta: float = 10.0
    d2: float = 0.001
    alpha: float = 0.1
    beta: float = 0.0
    epsilon: float = 0.0025
    support_radius: float = 0.1
    box_length: float = 1.0
    grid_size: int = 101
    dt: float = 0.01
    final_time: float = 4.0
    save_every: int = 100
    output_dir: str = "runs/fdm_101"
    enable_x64: bool = False

    @property
    def dx(self) -> float:
        return self.box_length / self.grid_size

    @property
    def steps(self) -> int:
        raw_steps = self.final_time / self.dt
        steps = int(round(raw_steps))
        if not math.isclose(raw_steps, steps, rel_tol=0.0, abs_tol=1.0e-10):
            raise ValueError("final_time must be an integer multiple of dt")
        return steps

    def validate(self) -> None:
        if min(self.d1, self.gamma, self.eta, self.d2, self.alpha, self.beta) < 0.0:
            raise ValueError("HAD parameters must be nonnegative")
        if self.epsilon <= 0.0 or self.support_radius <= 0.0:
            raise ValueError("epsilon and support_radius must be positive")
        if self.box_length <= 0.0 or self.grid_size < 3:
            raise ValueError("box_length must be positive and grid_size at least 3")
        if self.dt <= 0.0 or self.final_time <= 0.0 or self.save_every < 1:
            raise ValueError("dt, final_time, and save_every must be positive")
        max_diffusion = max(self.d1, self.d2)
        if max_diffusion > 0.0:
            diffusion_number = 6.0 * max_diffusion * self.dt / self.dx**2
            if diffusion_number > 1.0 + 1.0e-12:
                raise ValueError(
                    "explicit 3D diffusion is unstable: "
                    f"6*d*dt/dx^2={diffusion_number:.3g} > 1"
                )
        _ = self.steps

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "FDMConfig":
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "FDMConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


class FDMState(NamedTuple):
    rho: object
    f: object
    m: object
    step: object


def _add_face_divergence(output, flux, axis: int, dx: float):
    lower = [slice(None)] * 3
    upper = [slice(None)] * 3
    lower[axis] = slice(0, -1)
    upper[axis] = slice(1, None)
    output = output.at[tuple(lower)].add(flux / dx)
    output = output.at[tuple(upper)].add(-flux / dx)
    return output


class FDMSolver:
    """Cell-centered central-flux FDM with homogeneous zero total flux."""

    def __init__(self, config: FDMConfig):
        config.validate()
        jax.config.update("jax_enable_x64", config.enable_x64)
        self.config = config
        self._step_jit = jax.jit(self._step_core)
        self._advance_jit = jax.jit(self._advance_core, static_argnums=(1,))
        self._initial_rho_mass = None
        self._initial_m_mass = None

    def initial_state(self) -> FDMState:
        config = self.config
        dtype = jnp.float64 if config.enable_x64 else jnp.float32
        coordinates = (
            (jnp.arange(config.grid_size, dtype=dtype) + 0.5) * config.dx
            - 0.5 * config.box_length
        )
        radius_squared = (
            coordinates[:, None, None] ** 2
            + coordinates[None, :, None] ** 2
            + coordinates[None, None, :] ** 2
        )
        rho = jnp.exp(-radius_squared / config.epsilon) * (
            radius_squared <= config.support_radius**2
        )
        f = 1.0 - 0.5 * rho
        m = 0.5 * rho
        self._initial_rho_mass = float(
            np.asarray(jax.device_get(jnp.sum(rho))) * config.dx**3
        )
        self._initial_m_mass = 0.5 * self._initial_rho_mass
        return FDMState(
            rho=rho,
            f=f,
            m=m,
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def _rho_rhs(self, rho, f):
        config = self.config
        divergence = jnp.zeros_like(rho)
        for axis in range(3):
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = slice(0, -1)
            upper[axis] = slice(1, None)
            lower = tuple(lower)
            upper = tuple(upper)
            rho_gradient = (rho[upper] - rho[lower]) / config.dx
            f_gradient = (f[upper] - f[lower]) / config.dx
            rho_face = 0.5 * (rho[upper] + rho[lower])
            flux = config.d1 * rho_gradient - config.gamma * rho_face * f_gradient
            divergence = _add_face_divergence(
                divergence, flux, axis, config.dx
            )
        return divergence

    def _m_rhs(self, m):
        config = self.config
        divergence = jnp.zeros_like(m)
        for axis in range(3):
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = slice(0, -1)
            upper[axis] = slice(1, None)
            lower = tuple(lower)
            upper = tuple(upper)
            flux = config.d2 * (m[upper] - m[lower]) / config.dx
            divergence = _add_face_divergence(
                divergence, flux, axis, config.dx
            )
        return divergence

    def _step_core(self, state: FDMState) -> FDMState:
        config = self.config
        next_rho = state.rho + config.dt * self._rho_rhs(state.rho, state.f)
        next_f = state.f - config.eta * config.dt * state.m * state.f
        next_m = state.m + config.dt * (
            self._m_rhs(state.m) - config.beta * state.m + config.alpha * next_rho
        )
        return FDMState(
            rho=next_rho,
            f=next_f,
            m=next_m,
            step=state.step + 1,
        )

    def _advance_core(self, state: FDMState, count: int) -> FDMState:
        def body(_, current_state):
            return self._step_core(current_state)

        return lax.fori_loop(0, count, body, state)

    def step(self, state: FDMState) -> FDMState:
        return self._step_jit(state)

    def advance(self, state: FDMState, count: int) -> FDMState:
        if count < 1:
            return state
        return self._advance_jit(state, count)

    def diagnostics(self, state: FDMState) -> Dict[str, float]:
        if self._initial_rho_mass is None or self._initial_m_mass is None:
            raise RuntimeError("initial_state must be called before diagnostics")
        config = self.config
        rho, f, m, step_value = jax.device_get(state)
        step = int(np.asarray(step_value))
        cell_volume = config.dx**3
        rho_mass = float(np.sum(rho) * cell_volume)
        m_mass = float(np.sum(m) * cell_volume)
        if config.beta == 0.0:
            expected_m_mass = (
                self._initial_m_mass
                + config.alpha * self._initial_rho_mass * step * config.dt
            )
        else:
            factor = 1.0 - config.beta * config.dt
            expected_m_mass = (
                factor**step * self._initial_m_mass
                + config.alpha
                * self._initial_rho_mass
                * config.dt
                * (1.0 - factor**step)
                / (1.0 - factor)
            )
        return {
            "step": float(step),
            "time": step * config.dt,
            "rho_mass": rho_mass,
            "rho_mass_relative_error": abs(rho_mass - self._initial_rho_mass)
            / max(abs(self._initial_rho_mass), 1.0e-15),
            "m_integral": m_mass,
            "m_integral_expected": expected_m_mass,
            "m_integral_relative_error": abs(m_mass - expected_m_mass)
            / max(abs(expected_m_mass), 1.0e-15),
            "rho_min": float(np.min(rho)),
            "rho_max": float(np.max(rho)),
            "f_min": float(np.min(f)),
            "m_min": float(np.min(m)),
        }


def _write_json(path: Path, values: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_fdm_snapshot(output_dir: Path, config: FDMConfig, state: FDMState) -> Path:
    rho, f, m, step_value = jax.device_get(state)
    step = int(np.asarray(step_value))
    path = output_dir / f"fdm_snapshot_{step:06d}.npz"
    np.savez_compressed(
        path,
        step=np.asarray(step),
        time=np.asarray(step * config.dt),
        box_length=np.asarray(config.box_length),
        dx=np.asarray(config.dx),
        coordinate_convention=np.asarray("cell_centered"),
        rho=np.asarray(rho),
        f=np.asarray(f),
        m=np.asarray(m),
    )
    return path


def run_fdm(
    config: FDMConfig, log: Callable[[str], None] = print
) -> Dict[str, Any]:
    """Run a 3D FDM case and save restart-independent NPZ snapshots."""
    start = time.perf_counter()
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", config.to_dict())
    solver = FDMSolver(config)
    state = solver.initial_state()
    save_fdm_snapshot(output_dir, config, state)

    log(
        f"3D FDM: N={config.grid_size}^3, dx={config.dx:.6g}, "
        f"dt={config.dt:g}, steps={config.steps}, backend={jax.default_backend()}"
    )
    first_block_seconds = None
    completed = 0
    while completed < config.steps:
        block_steps = min(config.save_every, config.steps - completed)
        block_start = time.perf_counter()
        state = solver.advance(state, block_steps)
        state.rho.block_until_ready()
        block_seconds = time.perf_counter() - block_start
        if first_block_seconds is None:
            first_block_seconds = block_seconds
        completed += block_steps
        snapshot = save_fdm_snapshot(output_dir, config, state)
        diagnostics = solver.diagnostics(state)
        log(
            f"t={diagnostics['time']:.4g} | step={completed}/{config.steps} | "
            f"block={block_seconds:.3f}s | "
            f"rho mass error={diagnostics['rho_mass_relative_error']:.2e} | "
            f"rho min={diagnostics['rho_min']:.3e} | {snapshot.name}"
        )

    summary: Dict[str, Any] = {
        "method": "cell-centered conservative central-flux FDM",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "wall_seconds_including_compilation_and_output": time.perf_counter() - start,
        "first_block_seconds_including_compilation": first_block_seconds,
        "completed_steps": config.steps,
        "final_diagnostics": solver.diagnostics(state),
    }
    _write_json(output_dir / "summary.json", summary)
    log(
        "FDM finished in "
        f"{summary['wall_seconds_including_compilation_and_output']:.3f}s; "
        f"output: {output_dir}"
    )
    return summary
