# Cancer SIPF in JAX

This repository contains a GPU-capable implementation of the stochastic
interacting particle-field (SIPF) method for the three-dimensional
haptotaxis advection-diffusion (HAD) system

```text
rho_t = d1 Delta rho - gamma div(rho grad f)
f_t   = -eta m f
m_t   = d2 Delta m - beta m + alpha rho.
```

The implementation follows the update order in Section 3 of the paper:

1. Move the cell particles with Euler-Maruyama using `grad(f_n)`.
2. Update the ECM field with the explicit valid-mode convolution of `m_n f_n`.
3. Update the MDE field with implicit Euler using the new particles
   `rho_(n+1)`.

The particle density is never rasterized for the numerical update. Its retained
Fourier coefficients are evaluated directly from the empirical measure.
Rasterized `rho_grid` arrays are output only for visualization.

## Install

Create a clean environment from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[plot,test]"
```

The command above installs CPU JAX where appropriate. On a Linux machine with
an NVIDIA V100, install the CUDA 12 JAX wheel first:

```bash
python -m pip install --upgrade "jax[cuda12]"
python -m pip install -e ".[plot,test]"
```

JAX selects an available GPU automatically. Check the selected backend in the
first line printed by each run. Current platform-specific installation details
are maintained in the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Quick Start

Run a four-step smoke simulation:

```bash
cancer-sipf --config configs/quickstart.json
```

The same command works without installing the console script:

```bash
PYTHONPATH=src python -m cancer_sipf --config configs/quickstart.json
```

Plot the final snapshot:

```bash
cancer-sipf-plot runs/quickstart/snapshot_000004.npz
```

Run the radial experiment at the scale stated in the paper:

```bash
cancer-sipf --config configs/paper_radial.json
```

Run the two-cluster experiment:

```bash
cancer-sipf --config configs/paper_two_clusters.json
```

Run a short FDM/radial comparison check:

```bash
cancer-fdm-comparison --config configs/fdm_comparison_quick.json
```

Run the paper's `21^3` through `101^3` FDM grid sweep, the 801-cell radial
reference, and the radial SIPF case:

```bash
cancer-fdm-comparison --config configs/fdm_comparison.json
```

The comparison produces a CSV table of wall times and relative MDE errors,
radial profile files, and a final profile plot. The precise discretizations and
the distinction between the documented reproduction and the unavailable
original FDM source are described in
[docs/fdm_comparison.md](docs/fdm_comparison.md).

The first saved block includes JAX compilation time. Later blocks reflect
steady-state execution more closely. Reported `wall_seconds_including_compilation`
is elapsed wall-clock time for the complete run.

## Outputs

Each run directory contains:

- `config.json`: the exact configuration used;
- `snapshot_*.npz`: centered particle positions, `rho`, `f`, and `m` grids,
  and the Fourier coefficients of `f` and `m`;
- `summary.json`: backend, device, versions, timing, and the MDE mass check.

The particle positions in output files use the paper's centered box
`[-L/2,L/2)^3`. Internally they are shifted to `[0,L)^3`, so the phase induced
by the shift is handled once and is equivalent to the alternating Fourier
factor described in the paper.

## Performance Notes

- Particle work is split into fixed-size batches to avoid allocating a
  `P x H^3` phase matrix.
- Separable Fourier contractions reduce exponential evaluations from
  `O(P H^3)` to `O(P H)` while retaining the direct empirical coefficients.
- The ECM product uses zero-padded FFTs to reproduce the valid retained-mode
  convolution without circular aliasing.
- Multiple time steps between snapshots execute in one JIT-compiled JAX loop.
- `float32` is the default for GPU throughput. Set `enable_x64` to `true` for
  verification runs when the selected JAX backend supports it efficiently.

`particle_batch_size=256` is a conservative V100 starting point. Values from
128 to 1024 are worth benchmarking for a particular GPU and memory budget.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The tests cover the dealiased spectral product, real Fourier symmetry,
periodic particle movement, conservative FDM and radial mass identities,
radial averaging, and the discrete MDE mass identity.

## Numerical Conventions

See [docs/algorithm.md](docs/algorithm.md) for the exact coefficient
normalization, coordinate shift, splitting order, and Nyquist convention.

Before making the repository public, add the license agreed upon by all
authors and update the package metadata accordingly.
