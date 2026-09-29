"""Run from the repository root with: python experiments/run_fdm_comparison.py."""

from cancer_sipf.comparison import ComparisonConfig, run_comparison


if __name__ == "__main__":
    run_comparison(ComparisonConfig.from_json("configs/fdm_comparison.json"))
