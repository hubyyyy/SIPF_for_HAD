"""GPU-capable JAX solver for the three-step SIPF update."""

from __future__ import annotations

from typing import Dict, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random

from .config import SimulationConfig
from .initial_conditions import (
    initial_field_coefficients,
    sample_initial_particles,
    total_cell_mass,
)
from .spectral import (
    SpectralGrid,
    build_spectral_grid,
    coefficients_to_grid,
    enforce_real_spectrum,
    galerkin_product,
)


class SimulationState(NamedTuple):
    positions: object
    f_hat: object
    m_hat: object
    key: object
    step: object


class SIPFSolver:
    """SIPF discretization of the haptotaxis advection-diffusion system."""

    def __init__(self, config: SimulationConfig):
        config.validate()
        jax.config.update("jax_enable_x64", config.enable_x64)
        self.config = config
        self.grid = build_spectral_grid(
            config.modes, config.box_length, config.dealias_factor
        )
        self.mass = total_cell_mass(config)
        self._complex_dtype = jnp.complex128 if config.enable_x64 else jnp.complex64
        self._step_jit = jax.jit(self._step_core)
        self._advance_jit = jax.jit(self._advance_core, static_argnums=(1,))
        self._field_grid_jit = jax.jit(coefficients_to_grid)

    def initial_state(self) -> SimulationState:
        positions = sample_initial_particles(self.config)
        f_hat, m_hat = initial_field_coefficients(self.config, self.grid)
        return SimulationState(
            positions=positions,
            f_hat=enforce_real_spectrum(f_hat),
            m_hat=enforce_real_spectrum(m_hat),
            key=random.PRNGKey(self.config.seed + 1),
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def centered_positions(self, positions):
        return positions - 0.5 * self.config.box_length

    def field_on_grid(self, coefficients):
        return self._field_grid_jit(coefficients)

    def _gradient_at_particles(self, positions, f_hat):
        """Evaluate grad(f) by batched separable Fourier contractions."""
        batch_size = self.config.particle_batch_size
        particle_count = positions.shape[0]
        padded_count = ((particle_count + batch_size - 1) // batch_size) * batch_size
        padded = jnp.pad(positions, ((0, padded_count - particle_count), (0, 0)))
        chunks = padded.reshape((-1, batch_size, 3))
        wave_numbers = self.grid.wave_numbers
        kx = wave_numbers[:, None, None]
        ky = wave_numbers[None, :, None]
        kz = wave_numbers[None, None, :]
        gradient_coefficients = jnp.stack(
            (1j * kx * f_hat, 1j * ky * f_hat, 1j * kz * f_hat), axis=0
        )

        def evaluate_chunk(dummy, chunk):
            exp_x = jnp.exp(1j * chunk[:, 0, None] * wave_numbers[None, :])
            exp_y = jnp.exp(1j * chunk[:, 1, None] * wave_numbers[None, :])
            exp_z = jnp.exp(1j * chunk[:, 2, None] * wave_numbers[None, :])
            partial_xy = jnp.einsum(
                "pi,cijk->pcjk", exp_x, gradient_coefficients, optimize=True
            )
            partial_z = jnp.einsum(
                "pj,pcjk->pck", exp_y, partial_xy, optimize=True
            )
            values = jnp.einsum("pk,pck->pc", exp_z, partial_z, optimize=True)
            return dummy, jnp.real(values)

        _, gradients = lax.scan(
            evaluate_chunk, jnp.asarray(0, dtype=jnp.int32), chunks
        )
        return gradients.reshape((-1, 3))[:particle_count]

    def _particle_density_coefficients(self, positions):
        """Evaluate all retained empirical-measure Fourier coefficients."""
        batch_size = self.config.particle_batch_size
        particle_count = positions.shape[0]
        padded_count = ((particle_count + batch_size - 1) // batch_size) * batch_size
        padded = jnp.pad(positions, ((0, padded_count - particle_count), (0, 0)))
        chunks = padded.reshape((-1, batch_size, 3))
        mask = (
            (jnp.arange(padded_count) < particle_count)
            .astype(positions.dtype)
            .reshape((-1, batch_size))
        )
        wave_numbers = self.grid.wave_numbers
        initial = jnp.zeros(
            (self.config.modes,) * 3, dtype=self._complex_dtype
        )

        def accumulate(coefficients, values):
            chunk, active = values
            exp_x = jnp.exp(-1j * chunk[:, 0, None] * wave_numbers[None, :])
            exp_y = jnp.exp(-1j * chunk[:, 1, None] * wave_numbers[None, :])
            exp_z = jnp.exp(-1j * chunk[:, 2, None] * wave_numbers[None, :])
            partial_xy = jnp.einsum(
                "pi,pj,p->pij", exp_x, exp_y, active, optimize=True
            )
            contribution = jnp.einsum(
                "pij,pk->ijk", partial_xy, exp_z, optimize=True
            )
            return coefficients + contribution, None

        summed, _ = lax.scan(accumulate, initial, (chunks, mask))
        rho_hat = (self.mass / particle_count) * summed
        return enforce_real_spectrum(rho_hat)

    def _step_core(self, state: SimulationState) -> SimulationState:
        config = self.config
        next_key, noise_key = random.split(state.key)

        # Step 1: rho_n -> rho_{n+1}, using grad(f_n).
        gradient = self._gradient_at_particles(state.positions, state.f_hat)
        noise = random.normal(
            noise_key, state.positions.shape, dtype=state.positions.dtype
        )
        next_positions = (
            state.positions
            + config.gamma * config.dt * gradient
            + jnp.sqrt(2.0 * config.d1 * config.dt) * noise
        )
        next_positions = jnp.mod(next_positions, config.box_length)

        # Step 2: f_n -> f_{n+1}, using the valid-mode convolution m_n f_n.
        mf_hat = galerkin_product(state.m_hat, state.f_hat, self.grid)
        next_f_hat = enforce_real_spectrum(
            state.f_hat - config.eta * config.dt * mf_hat
        )

        # Step 3: m_n -> m_{n+1}, using rho_{n+1} and implicit Euler.
        rho_hat = self._particle_density_coefficients(next_positions)
        denominator = 1.0 + config.dt * (
            config.d2 * self.grid.wave_number_squared + config.beta
        )
        next_m_hat = enforce_real_spectrum(
            (state.m_hat + config.alpha * config.dt * rho_hat) / denominator
        )

        return SimulationState(
            positions=next_positions,
            f_hat=next_f_hat,
            m_hat=next_m_hat,
            key=next_key,
            step=state.step + 1,
        )

    def _advance_core(self, state: SimulationState, count: int) -> SimulationState:
        def body(_, current_state):
            return self._step_core(current_state)

        return lax.fori_loop(0, count, body, state)

    def step(self, state: SimulationState) -> SimulationState:
        return self._step_jit(state)

    def advance(self, state: SimulationState, count: int) -> SimulationState:
        if count < 1:
            return state
        return self._advance_jit(state, count)

    def diagnostics(self, state: SimulationState) -> Dict[str, float]:
        zero_mode_m = np.asarray(jax.device_get(state.m_hat[0, 0, 0])).real
        zero_mode_f = np.asarray(jax.device_get(state.f_hat[0, 0, 0])).real
        step = int(np.asarray(jax.device_get(state.step)))
        m_integral = float(zero_mode_m * self.config.box_length**3)
        f_integral = float(zero_mode_f * self.config.box_length**3)
        initial_m_integral = 0.5 * self.mass
        if self.config.beta == 0.0:
            expected_m_integral = (
                initial_m_integral
                + self.config.alpha * self.mass * step * self.config.dt
            )
        else:
            q = 1.0 / (1.0 + self.config.beta * self.config.dt)
            expected_m_integral = (
                q**step * initial_m_integral
                + self.config.alpha
                * self.mass
                * self.config.dt
                * q
                * (1.0 - q**step)
                / (1.0 - q)
            )
        relative_m_error = abs(m_integral - expected_m_integral) / max(
            abs(expected_m_integral), 1.0e-15
        )
        return {
            "step": float(step),
            "time": step * self.config.dt,
            "cell_mass": self.mass,
            "m_integral": m_integral,
            "m_integral_expected": expected_m_integral,
            "m_integral_relative_error": relative_m_error,
            "f_integral": f_integral,
        }
