"""High-resolution spherical reference solver for the radial HAD system."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Union

import numpy as np
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class RadialConfig:
    """Configuration for the one-dimensional spherical reference solution."""

    d1: float = 0.001
    gamma: float = 0.005
    eta: float = 10.0
    d2: float = 0.001
    alpha: float = 0.1
    beta: float = 0.0
    epsilon: float = 0.0025
    support_radius: float = 0.1
    domain_radius: float = 1.0
    radial_cells: int = 801
    dt: float = 0.001
    final_time: float = 4.0
    save_every: int = 1000
    output_dir: str = "runs/radial_reference"

    @property
    def dr(self) -> float:
        return self.domain_radius / self.radial_cells

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
        if self.domain_radius <= self.support_radius:
            raise ValueError("domain_radius must exceed support_radius")
        if self.radial_cells < 3:
            raise ValueError("radial_cells must be at least 3")
        if self.dt <= 0.0 or self.final_time <= 0.0 or self.save_every < 1:
            raise ValueError("dt, final_time, and save_every must be positive")
        _ = self.steps

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "RadialConfig":
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "RadialConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


class RadialState(NamedTuple):
    rho: np.ndarray
    f: np.ndarray
    m: np.ndarray
    step: int


def _radial_laplacian_coefficients(config: RadialConfig):
    faces = np.arange(config.radial_cells + 1, dtype=np.float64) * config.dr
    areas = faces**2
    volumes = (faces[1:] ** 3 - faces[:-1] ** 3) / 3.0
    lower = np.zeros(config.radial_cells, dtype=np.float64)
    upper = np.zeros(config.radial_cells, dtype=np.float64)
    lower[1:] = areas[1:-1] / (config.dr * volumes[1:])
    upper[:-1] = areas[1:-1] / (config.dr * volumes[:-1])
    diagonal = -(lower + upper)
    return faces, areas, volumes, lower, diagonal, upper


def _implicit_banded_matrix(
    diffusion: float,
    reaction: float,
    dt: float,
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    size = diagonal.size
    matrix = np.zeros((3, size), dtype=np.float64)
    matrix[0, 1:] = -dt * diffusion * upper[:-1]
    matrix[1, :] = 1.0 - dt * diffusion * diagonal + dt * reaction
    matrix[2, :-1] = -dt * diffusion * lower[1:]
    return matrix


class RadialSolver:
    """Finite-volume radial solver with implicit diffusion and zero face flux."""

    def __init__(self, config: RadialConfig):
        config.validate()
        self.config = config
        (
            self.faces,
            self.face_areas,
            self.shell_volumes_without_4pi,
            lower,
            diagonal,
            upper,
        ) = _radial_laplacian_coefficients(config)
        self.radii = (np.arange(config.radial_cells) + 0.5) * config.dr
        self._rho_matrix = _implicit_banded_matrix(
            config.d1, 0.0, config.dt, lower, diagonal, upper
        )
        self._m_matrix = _implicit_banded_matrix(
            config.d2, config.beta, config.dt, lower, diagonal, upper
        )
        self._initial_rho_mass = None
        self._initial_m_mass = None

    @property
    def shell_volumes(self):
        return 4.0 * np.pi * self.shell_volumes_without_4pi

    def initial_state(self) -> RadialState:
        config = self.config
        rho = np.exp(-(self.radii**2) / config.epsilon) * (
            self.radii <= config.support_radius
        )
        f = 1.0 - 0.5 * rho
        m = 0.5 * rho
        self._initial_rho_mass = float(np.dot(rho, self.shell_volumes))
        self._initial_m_mass = float(np.dot(m, self.shell_volumes))
        return RadialState(rho=rho, f=f, m=m, step=0)

    def _haptotaxis_divergence(self, rho, f):
        config = self.config
        flux = np.zeros(config.radial_cells + 1, dtype=np.float64)
        rho_face = 0.5 * (rho[1:] + rho[:-1])
        f_gradient = (f[1:] - f[:-1]) / config.dr
        flux[1:-1] = -config.gamma * rho_face * f_gradient
        return (
            self.face_areas[1:] * flux[1:]
            - self.face_areas[:-1] * flux[:-1]
        ) / self.shell_volumes_without_4pi

    def step(self, state: RadialState) -> RadialState:
        config = self.config
        rho_rhs = state.rho + config.dt * self._haptotaxis_divergence(
            state.rho, state.f
        )
        next_rho = solve_banded(
            (1, 1), self._rho_matrix, rho_rhs, overwrite_b=False, check_finite=False
        )
        next_f = state.f - config.eta * config.dt * state.m * state.f
        m_rhs = state.m + config.alpha * config.dt * next_rho
        next_m = solve_banded(
            (1, 1), self._m_matrix, m_rhs, overwrite_b=False, check_finite=False
        )
        return RadialState(
            rho=next_rho,
            f=next_f,
            m=next_m,
            step=state.step + 1,
        )

    def advance(self, state: RadialState, count: int) -> RadialState:
        for _ in range(count):
            state = self.step(state)
        return state

    def diagnostics(self, state: RadialState) -> Dict[str, float]:
        if self._initial_rho_mass is None or self._initial_m_mass is None:
            raise RuntimeError("initial_state must be called before diagnostics")
        config = self.config
        rho_mass = float(np.dot(state.rho, self.shell_volumes))
        m_mass = float(np.dot(state.m, self.shell_volumes))
        if config.beta == 0.0:
            expected_m_mass = (
                self._initial_m_mass
                + config.alpha * self._initial_rho_mass * state.step * config.dt
            )
        else:
            factor = 1.0 / (1.0 + config.beta * config.dt)
            expected_m_mass = (
                factor**state.step * self._initial_m_mass
                + config.alpha
                * self._initial_rho_mass
                * config.dt
                * factor
                * (1.0 - factor**state.step)
                / (1.0 - factor)
            )
        return {
            "step": float(state.step),
            "time": state.step * config.dt,
            "rho_mass": rho_mass,
            "rho_mass_relative_error": abs(rho_mass - self._initial_rho_mass)
            / max(abs(self._initial_rho_mass), 1.0e-15),
            "m_integral": m_mass,
            "m_integral_expected": expected_m_mass,
            "m_integral_relative_error": abs(m_mass - expected_m_mass)
            / max(abs(expected_m_mass), 1.0e-15),
            "rho_min": float(np.min(state.rho)),
            "rho_max": float(np.max(state.rho)),
            "f_min": float(np.min(state.f)),
            "m_min": float(np.min(state.m)),
        }


def _write_json(path: Path, values: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_radial_snapshot(
    output_dir: Path,
    solver: RadialSolver,
    state: RadialState,
) -> Path:
    path = output_dir / f"radial_snapshot_{state.step:06d}.npz"
    np.savez_compressed(
        path,
        step=np.asarray(state.step),
        time=np.asarray(state.step * solver.config.dt),
        radius=solver.radii,
        shell_volume=solver.shell_volumes,
        rho=state.rho,
        f=state.f,
        m=state.m,
    )
    return path


def run_radial(
    config: RadialConfig, log: Callable[[str], None] = print
) -> Dict[str, Any]:
    """Run and save the high-resolution radial reference solution."""
    start = time.perf_counter()
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", config.to_dict())
    solver = RadialSolver(config)
    state = solver.initial_state()
    save_radial_snapshot(output_dir, solver, state)
    log(
        f"Radial reference: cells={config.radial_cells}, dr={config.dr:.6g}, "
        f"dt={config.dt:g}, steps={config.steps}"
    )

    completed = 0
    while completed < config.steps:
        block_steps = min(config.save_every, config.steps - completed)
        block_start = time.perf_counter()
        state = solver.advance(state, block_steps)
        block_seconds = time.perf_counter() - block_start
        completed += block_steps
        snapshot = save_radial_snapshot(output_dir, solver, state)
        diagnostics = solver.diagnostics(state)
        log(
            f"t={diagnostics['time']:.4g} | step={completed}/{config.steps} | "
            f"block={block_seconds:.3f}s | "
            f"rho mass error={diagnostics['rho_mass_relative_error']:.2e} | "
            f"{snapshot.name}"
        )

    summary: Dict[str, Any] = {
        "method": "spherical finite volume with implicit diffusion",
        "wall_seconds_including_output": time.perf_counter() - start,
        "completed_steps": config.steps,
        "final_diagnostics": solver.diagnostics(state),
    }
    _write_json(output_dir / "summary.json", summary)
    log(
        f"Radial reference finished in {summary['wall_seconds_including_output']:.3f}s; "
        f"output: {output_dir}"
    )
    return summary
