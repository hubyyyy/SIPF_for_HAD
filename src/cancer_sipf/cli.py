"""Command-line interfaces for running and plotting simulations."""

from __future__ import annotations

import argparse
from dataclasses import replace

from .config import SimulationConfig


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the JAX SIPF solver for the 3D cancer-cell HAD model."
    )
    parser.add_argument("--config", required=True, help="Path to a JSON configuration")
    parser.add_argument("--output", help="Override output_dir from the configuration")
    parser.add_argument(
        "--final-time", type=float, help="Override final_time from the configuration"
    )
    parser.add_argument("--seed", type=int, help="Override the random seed")
    parser.add_argument(
        "--check", action="store_true", help="Validate and print the configuration only"
    )
    return parser


def main(argv=None) -> int:
    args = build_run_parser().parse_args(argv)
    config = SimulationConfig.from_json(args.config)
    overrides = {}
    if args.output:
        overrides["output_dir"] = args.output
    if args.final_time is not None:
        overrides["final_time"] = args.final_time
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        config = replace(config, **overrides)
        config.validate()
    if args.check:
        print(config.to_dict())
        return 0

    from .runner import run_simulation

    run_simulation(config)
    return 0


def plot_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Plot one SIPF NPZ snapshot")
    parser.add_argument("snapshot", help="Path to snapshot_*.npz")
    parser.add_argument("--output", help="Output PNG path")
    args = parser.parse_args(argv)

    from .output import plot_snapshot

    output_path = plot_snapshot(args.snapshot, args.output)
    print(output_path)
    return 0

