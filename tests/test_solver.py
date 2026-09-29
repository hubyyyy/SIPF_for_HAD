import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cancer_sipf.config import SimulationConfig
from cancer_sipf.solver import SIPFSolver


def small_config(**overrides):
    values = {
        "d1": 0.001,
        "gamma": 0.005,
        "eta": 10.0,
        "d2": 0.001,
        "alpha": 0.1,
        "beta": 0.0,
        "box_length": 1.0,
        "modes": 6,
        "particles": 24,
        "dt": 0.01,
        "final_time": 0.01,
        "particle_batch_size": 8,
        "dealias_factor": 2,
        "initial_grid_factor": 4,
        "initial_condition": "radial",
        "epsilon": 0.0025,
        "support_radius": 0.1,
        "centers": ((0.0, 0.0, 0.0),),
        "seed": 7,
        "save_every": 1,
        "output_dir": "runs/test",
        "enable_x64": False,
    }
    values.update(overrides)
    return SimulationConfig(**values)


class SolverTests(unittest.TestCase):
    def test_one_step_is_finite_periodic_and_mass_consistent(self):
        solver = SIPFSolver(small_config())
        state = solver.step(solver.initial_state())
        state.positions.block_until_ready()
        positions = np.asarray(jax.device_get(state.positions))
        self.assertTrue(np.isfinite(positions).all())
        self.assertGreaterEqual(positions.min(), 0.0)
        self.assertLess(positions.max(), solver.config.box_length)
        diagnostics = solver.diagnostics(state)
        self.assertLess(diagnostics["m_integral_relative_error"], 2.0e-6)

    def test_constant_ecm_has_zero_haptotactic_gradient(self):
        solver = SIPFSolver(small_config())
        state = solver.initial_state()
        constant = np.zeros((solver.config.modes,) * 3, dtype=np.complex64)
        constant[0, 0, 0] = 1.0
        gradient = np.asarray(
            jax.device_get(
                solver._gradient_at_particles(state.positions, constant)
            )
        )
        np.testing.assert_allclose(gradient, 0.0, atol=1.0e-7)

    def test_fourier_particle_operations_match_direct_formulas(self):
        solver = SIPFSolver(small_config())
        state = solver.initial_state()
        modes = solver.config.modes
        coefficients = jnp.zeros((modes,) * 3, dtype=jnp.complex64)
        coefficients = coefficients.at[1, 0, 0].set(0.5)
        coefficients = coefficients.at[-1, 0, 0].set(0.5)
        gradient = np.asarray(
            jax.device_get(
                solver._gradient_at_particles(state.positions, coefficients)
            )
        )
        positions = np.asarray(jax.device_get(state.positions))
        expected_x = -2.0 * np.pi * np.sin(2.0 * np.pi * positions[:, 0])
        np.testing.assert_allclose(gradient[:, 0], expected_x, atol=2.0e-5)
        np.testing.assert_allclose(gradient[:, 1:], 0.0, atol=1.0e-7)

        rho_hat = np.asarray(
            jax.device_get(
                solver._particle_density_coefficients(state.positions)
            )
        )
        expected_mode = solver.mass * np.mean(
            np.exp(-2j * np.pi * positions[:, 0])
        )
        np.testing.assert_allclose(rho_hat[1, 0, 0], expected_mode, atol=2.0e-7)


if __name__ == "__main__":
    unittest.main()
