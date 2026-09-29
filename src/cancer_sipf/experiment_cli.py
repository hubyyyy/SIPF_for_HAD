"""Command-line entry points for FDM and radial comparison experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


def fdm_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one 3D conservative FDM case")
    parser.add_argument("--config", required=True, help="FDM JSON configuration")
    args = parser.parse_args(argv)
    from .fdm import FDMConfig, run_fdm

    run_fdm(FDMConfig.from_json(args.config))
    return 0


def radial_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the spherical high-resolution HAD reference solver"
    )
    parser.add_argument("--config", required=True, help="Radial JSON configuration")
    args = parser.parse_args(argv)
    from .radial import RadialConfig, run_radial

    run_radial(RadialConfig.from_json(args.config))
    return 0


def radialize_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert an FDM or SIPF 3D snapshot into radial profiles"
    )
    parser.add_argument("snapshot", help="Input NPZ snapshot")
    parser.add_argument("--kind", choices=("fdm", "sipf"), required=True)
    parser.add_argument("--output", required=True, help="Output profile NPZ")
    parser.add_argument("--bins", type=int)
    parser.add_argument("--max-radius", type=float)
    parser.add_argument(
        "--total-mass",
        type=float,
        help="SIPF cell mass for the optional particle radial density",
    )
    args = parser.parse_args(argv)
    from .profiles import radialize_fdm_snapshot, radialize_sipf_snapshot

    if args.kind == "fdm":
        output = radialize_fdm_snapshot(
            args.snapshot, args.output, args.bins, args.max_radius
        )
    else:
        output = radialize_sipf_snapshot(
            args.snapshot,
            args.output,
            total_mass=args.total_mass,
            bins=args.bins,
            max_radius=args.max_radius,
        )
    print(Path(output))
    return 0


def comparison_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the radial reference, FDM grid sweep, and SIPF comparison"
    )
    parser.add_argument("--config", required=True, help="Comparison JSON configuration")
    args = parser.parse_args(argv)
    from .comparison import ComparisonConfig, run_comparison

    run_comparison(ComparisonConfig.from_json(args.config))
    return 0
