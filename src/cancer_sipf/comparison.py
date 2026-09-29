"""End-to-end radial-reference, FDM-grid, and optional SIPF comparison."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from .config import SimulationConfig
from .fdm import FDMConfig, run_fdm
from .profiles import (
    RadialProfile,
    radial_average_grid,
    radialize_sipf_snapshot,
    relative_l2_against_reference,
    save_profile,
)
from .radial import RadialConfig, run_radial
from .runner import run_simulation


@dataclass(frozen=True)
class ComparisonConfig:
    """Parameters for the paper's radial/FDM/SIPF comparison workflow."""

    d1: float = 0.001
    gamma: float = 0.005
    eta: float = 10.0
    d2: float = 0.001
    alpha: float = 0.1
    beta: float = 0.0
    epsilon: float = 0.0025
    support_radius: float = 0.1
    box_length: float = 1.0
    final_time: float = 4.0
    snapshot_interval: float = 1.0
    fdm_grid_sizes: Tuple[int, ...] = (21, 41, 61, 81, 101)
    fdm_dt: float = 0.01
    radial_cells: int = 801
    radial_domain_radius: float = 1.0
    radial_dt: float = 0.001
    profile_max_radius: float = 0.5
    profile_bins: int = 20
    run_sipf: bool = True
    sipf_config: str = "configs/paper_radial.json"
    sipf_run_dir: str = ""
    output_dir: str = "runs/fdm_comparison"
    enable_x64: bool = False

    def validate(self) -> None:
        if not self.fdm_grid_sizes or any(size < 3 for size in self.fdm_grid_sizes):
            raise ValueError("fdm_grid_sizes must contain integers of at least 3")
        if tuple(sorted(set(self.fdm_grid_sizes))) != self.fdm_grid_sizes:
            raise ValueError("fdm_grid_sizes must be strictly increasing")
        if min(self.final_time, self.snapshot_interval, self.fdm_dt, self.radial_dt) <= 0:
            raise ValueError("times and time steps must be positive")
        if self.profile_max_radius <= 0.0:
            raise ValueError("profile_max_radius must be positive")
        if self.profile_bins < 1:
            raise ValueError("profile_bins must be positive")
        for value, step, name in (
            (self.final_time, self.fdm_dt, "fdm_dt"),
            (self.final_time, self.radial_dt, "radial_dt"),
            (self.snapshot_interval, self.fdm_dt, "snapshot_interval/fdm_dt"),
            (
                self.snapshot_interval,
                self.radial_dt,
                "snapshot_interval/radial_dt",
            ),
        ):
            if not math.isclose(value / step, round(value / step), abs_tol=1.0e-10):
                raise ValueError(f"{name} must divide its target interval")

    def to_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        values["fdm_grid_sizes"] = list(self.fdm_grid_sizes)
        return values

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ComparisonConfig":
        data = dict(values)
        if "fdm_grid_sizes" in data:
            data["fdm_grid_sizes"] = tuple(int(v) for v in data["fdm_grid_sizes"])
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ComparisonConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _write_json(path: Path, values: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _latest_snapshot(directory: Path, pattern: str) -> Path:
    snapshots = sorted(directory.glob(pattern))
    if not snapshots:
        raise FileNotFoundError(f"no snapshots matching {pattern} in {directory}")
    return snapshots[-1]


def _plot_final_profiles(
    output_path: Path,
    reference_radius: np.ndarray,
    reference_m: np.ndarray,
    fdm_profiles,
    sipf_profile_path: Optional[Path],
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    mask = reference_radius <= 0.5
    axis.plot(
        reference_radius[mask],
        reference_m[mask],
        color="black",
        linewidth=2.0,
        label="Radial reference",
    )
    for grid_size, profile in fdm_profiles:
        valid = np.isfinite(profile.mean)
        axis.plot(
            profile.radius[valid],
            profile.mean[valid],
            linewidth=1.2,
            label=f"FDM {grid_size}^3",
        )
    if sipf_profile_path is not None:
        with np.load(sipf_profile_path) as profile:
            valid = np.isfinite(profile["m_mean"])
            axis.plot(
                profile["m_radius"][valid],
                profile["m_mean"][valid],
                "o-",
                markersize=3,
                linewidth=1.2,
                label="SIPF",
            )
    axis.set_xlabel("r")
    axis.set_ylabel("m(r,T)")
    axis.set_xlim(0.0, 0.5)
    axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_comparison(
    config: ComparisonConfig, log: Callable[[str], None] = print
) -> Dict[str, Any]:
    """Run the complete grid sweep and compare final radial MDE profiles."""
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", config.to_dict())

    radial_output = output_dir / "radial_reference"
    radial_config = RadialConfig(
        d1=config.d1,
        gamma=config.gamma,
        eta=config.eta,
        d2=config.d2,
        alpha=config.alpha,
        beta=config.beta,
        epsilon=config.epsilon,
        support_radius=config.support_radius,
        domain_radius=config.radial_domain_radius,
        radial_cells=config.radial_cells,
        dt=config.radial_dt,
        final_time=config.final_time,
        save_every=int(round(config.snapshot_interval / config.radial_dt)),
        output_dir=str(radial_output),
    )
    radial_summary = run_radial(radial_config, log)
    radial_snapshot = _latest_snapshot(radial_output, "radial_snapshot_*.npz")
    with np.load(radial_snapshot) as reference:
        reference_radius = np.asarray(reference["radius"])
        reference_m = np.asarray(reference["m"])

    rows = []
    fdm_profiles = []
    previous = None
    profile_edges = np.linspace(
        0.0, config.profile_max_radius, config.profile_bins + 1
    )
    for grid_size in config.fdm_grid_sizes:
        case_output = output_dir / f"fdm_{grid_size}"
        fdm_config = FDMConfig(
            d1=config.d1,
            gamma=config.gamma,
            eta=config.eta,
            d2=config.d2,
            alpha=config.alpha,
            beta=config.beta,
            epsilon=config.epsilon,
            support_radius=config.support_radius,
            box_length=config.box_length,
            grid_size=grid_size,
            dt=config.fdm_dt,
            final_time=config.final_time,
            save_every=int(round(config.snapshot_interval / config.fdm_dt)),
            output_dir=str(case_output),
            enable_x64=config.enable_x64,
        )
        fdm_summary = run_fdm(fdm_config, log)
        final_snapshot = _latest_snapshot(case_output, "fdm_snapshot_*.npz")
        with np.load(final_snapshot) as snapshot:
            m_grid = np.asarray(snapshot["m"])
        profile = radial_average_grid(
            m_grid,
            config.box_length,
            bins=profile_edges,
            coordinate_convention="cell_centered",
            max_radius=config.profile_max_radius,
        )
        profile_path = output_dir / f"fdm_{grid_size}_m_profile.npz"
        save_profile(
            profile_path,
            profile,
            grid_size=np.asarray(grid_size),
            time=np.asarray(config.final_time),
        )
        fdm_profiles.append((grid_size, profile))
        error = relative_l2_against_reference(
            profile, reference_radius, reference_m
        )
        wall_seconds = float(
            fdm_summary["wall_seconds_including_compilation_and_output"]
        )
        row = {
            "grid_size": grid_size,
            "dx": config.box_length / grid_size,
            "wall_seconds": wall_seconds,
            "runtime_order": None,
            "relative_l2_error_m": error,
            "error_rate": None,
            "rho_mass_relative_error": fdm_summary["final_diagnostics"][
                "rho_mass_relative_error"
            ],
            "rho_min": fdm_summary["final_diagnostics"]["rho_min"],
        }
        if previous is not None:
            row["runtime_order"] = abs(
                math.log(previous["wall_seconds"] / wall_seconds)
                / math.log(previous["dx"] / row["dx"])
            )
            row["error_rate"] = abs(
                math.log(previous["relative_l2_error_m"] / error)
                / math.log(previous["dx"] / row["dx"])
            )
        rows.append(row)
        previous = row

    csv_path = output_dir / "fdm_grid_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sipf_result = None
    sipf_profile_path = None
    sipf_run_dir = Path(config.sipf_run_dir) if config.sipf_run_dir else None
    if config.run_sipf:
        sipf_run_dir = output_dir / "sipf"
        sipf_config = SimulationConfig.from_json(config.sipf_config)
        if not math.isclose(
            config.snapshot_interval / sipf_config.dt,
            round(config.snapshot_interval / sipf_config.dt),
            abs_tol=1.0e-10,
        ):
            raise ValueError("snapshot_interval must be a multiple of the SIPF dt")
        sipf_config = replace(
            sipf_config,
            final_time=config.final_time,
            save_every=int(round(config.snapshot_interval / sipf_config.dt)),
            output_dir=str(sipf_run_dir),
        )
        sipf_summary = run_simulation(sipf_config, log)
    else:
        sipf_summary = None

    if sipf_run_dir is not None:
        sipf_snapshot = _latest_snapshot(sipf_run_dir, "snapshot_*.npz")
        with np.load(sipf_snapshot) as snapshot:
            sipf_time = float(snapshot["time"])
        if not math.isclose(
            sipf_time, config.final_time, rel_tol=0.0, abs_tol=1.0e-10
        ):
            raise ValueError(
                f"SIPF snapshot time {sipf_time:g} does not match "
                f"comparison final_time {config.final_time:g}"
            )
        sipf_profile_path = output_dir / "sipf_m_profile.npz"
        if config.run_sipf:
            total_mass = sipf_summary["final_diagnostics"]["cell_mass"]
        else:
            with (sipf_run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                total_mass = json.load(handle)["final_diagnostics"]["cell_mass"]
        radialize_sipf_snapshot(
            sipf_snapshot,
            sipf_profile_path,
            total_mass=total_mass,
            bins=config.profile_bins,
            max_radius=config.profile_max_radius,
        )
        with np.load(sipf_profile_path) as profile:
            sipf_radial_profile = RadialProfile(
                radius=np.asarray(profile["m_radius"]),
                mean=np.asarray(profile["m_mean"]),
                count=np.asarray(profile["m_count"]),
                edges=np.asarray(profile["edges"]),
            )
            sipf_error = relative_l2_against_reference(
                sipf_radial_profile, reference_radius, reference_m
            )
        sipf_result = {
            "relative_l2_error_m": sipf_error,
            "run_directory": str(sipf_run_dir),
            "profile": str(sipf_profile_path),
        }

    plot_path = output_dir / "final_m_radial_profiles.png"
    _plot_final_profiles(
        plot_path,
        reference_radius,
        reference_m,
        fdm_profiles,
        sipf_profile_path,
    )
    summary = {
        "radial_reference": radial_summary,
        "fdm_grid_sweep": rows,
        "sipf": sipf_result,
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    _write_json(output_dir / "comparison_summary.json", summary)
    log(f"Comparison output: {output_dir}")
    return summary
