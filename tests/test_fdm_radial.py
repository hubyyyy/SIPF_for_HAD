import unittest

import numpy as np

from cancer_sipf.fdm import FDMConfig, FDMSolver
from cancer_sipf.profiles import radial_average_grid
from cancer_sipf.radial import RadialConfig, RadialSolver


class FDMAndRadialTests(unittest.TestCase):
    def test_fdm_conserves_cell_mass_and_mde_identity(self):
        config = FDMConfig(
            grid_size=9,
            dt=0.01,
            final_time=0.01,
            save_every=1,
            output_dir="runs/test_fdm",
        )
        solver = FDMSolver(config)
        state = solver.step(solver.initial_state())
        state.rho.block_until_ready()
        diagnostics = solver.diagnostics(state)
        self.assertLess(diagnostics["rho_mass_relative_error"], 2.0e-6)
        self.assertLess(diagnostics["m_integral_relative_error"], 2.0e-6)

    def test_radial_solver_conserves_cell_mass_and_mde_identity(self):
        config = RadialConfig(
            radial_cells=101,
            dt=0.001,
            final_time=0.002,
            save_every=1,
            output_dir="runs/test_radial",
        )
        solver = RadialSolver(config)
        state = solver.advance(solver.initial_state(), 2)
        diagnostics = solver.diagnostics(state)
        self.assertLess(diagnostics["rho_mass_relative_error"], 2.0e-12)
        self.assertLess(diagnostics["m_integral_relative_error"], 2.0e-12)
        self.assertTrue(np.isfinite(state.rho).all())
        self.assertTrue(np.isfinite(state.m).all())

    def test_radial_average_preserves_constant_fields(self):
        values = np.full((11, 11, 11), 2.5)
        for convention in ("cell_centered", "fft"):
            profile = radial_average_grid(
                values,
                box_length=1.0,
                bins=5,
                coordinate_convention=convention,
                max_radius=0.5,
            )
            nonempty = profile.count > 0
            np.testing.assert_allclose(profile.mean[nonempty], 2.5)


if __name__ == "__main__":
    unittest.main()
