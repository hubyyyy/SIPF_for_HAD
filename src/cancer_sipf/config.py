"""Configuration objects for SIPF simulations."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union


@dataclass(frozen=True)
class SimulationConfig:
    """Physical parameters, discretization choices, and initial data."""

    # HAD model parameters
    d1: float = 0.001
    gamma: float = 0.005
    eta: float = 10.0
    d2: float = 0.001
    alpha: float = 0.1
    beta: float = 0.0

    # Time and spectral discretization
    box_length: float = 1.0
    modes: int = 24
    particles: int = 10_000
    dt: float = 0.01
    final_time: float = 4.0
    particle_batch_size: int = 256
    dealias_factor: int = 2
    initial_grid_factor: int = 4

    # Initial data from the paper
    initial_condition: str = "radial"
    epsilon: float = 0.0025
    support_radius: float = 0.1
    centers: Tuple[Tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)

    # Reproducibility and output
    seed: int = 123
    save_every: int = 100
    output_dir: str = "runs/paper_radial"
    enable_x64: bool = False

    @property
    def steps(self) -> int:
        raw_steps = self.final_time / self.dt
        steps = int(round(raw_steps))
        if not math.isclose(raw_steps, steps, rel_tol=0.0, abs_tol=1.0e-10):
            raise ValueError("final_time must be an integer multiple of dt")
        return steps

    def validate(self) -> None:
        if self.d1 < 0.0 or self.d2 < 0.0:
            raise ValueError("diffusion coefficients d1 and d2 must be nonnegative")
        if min(self.gamma, self.eta, self.alpha, self.beta) < 0.0:
            raise ValueError("gamma, eta, alpha, and beta must be nonnegative")
        if self.box_length <= 0.0 or self.dt <= 0.0 or self.final_time <= 0.0:
            raise ValueError("box_length, dt, and final_time must be positive")
        if self.modes < 4 or self.modes % 2:
            raise ValueError("modes must be an even integer of at least 4")
        if self.particles < 1 or self.particle_batch_size < 1:
            raise ValueError("particles and particle_batch_size must be positive")
        if self.dealias_factor < 2:
            raise ValueError("dealias_factor must be at least 2")
        if self.initial_grid_factor < 2:
            raise ValueError("initial_grid_factor must be at least 2")
        if self.initial_condition not in {"radial", "two_clusters"}:
            raise ValueError("initial_condition must be 'radial' or 'two_clusters'")
        expected_centers = 1 if self.initial_condition == "radial" else 2
        if len(self.centers) != expected_centers:
            raise ValueError(
                f"{self.initial_condition} requires {expected_centers} center(s)"
            )
        if self.epsilon <= 0.0 or self.support_radius <= 0.0:
            raise ValueError("epsilon and support_radius must be positive")
        half_box = 0.5 * self.box_length
        if any(abs(value) + self.support_radius >= half_box for c in self.centers for value in c):
            raise ValueError("each initial cluster must lie strictly inside the box")
        if self.save_every < 1:
            raise ValueError("save_every must be positive")
        _ = self.steps

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["centers"] = [list(center) for center in self.centers]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        values = dict(data)
        if "centers" in values:
            values["centers"] = tuple(tuple(float(v) for v in c) for c in values["centers"])
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "SimulationConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
