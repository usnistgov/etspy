"""Tests for the projection matching alignment sub-module of ETSpy."""

from typing import cast

import astra
import numpy as np
import pytest
from scipy import ndimage
from scipy.stats import pearsonr

import etspy.api as etspy
from etspy import projmatch
from etspy import simulation as sim

RMSE_THRESH = 0.1
CORR_COEFF = 0.95
NOISE_THRESH = 0.1


@pytest.fixture(scope="class")
def aligned_simulation():
    """Create a simluated misaligned stack with kwown shifts for use with tests."""
    np.random.seed(42)

    # Create the simulation object
    obj = np.zeros([100, 100])
    obj[70:80, 70:80] = 10
    obj[30:65, 30:65] = 8
    obj[15:25, 15:25] = 5
    obj = ndimage.gaussian_filter(obj, sigma=1)

    # Create the simulated tilt series
    sino = sim.create_model_tilt_series(
        obj[np.newaxis, :, :],
        angles=np.linspace(0, 180, 90),
    )
    shifted = sim.misalign_stack(sino, min_shift=-2, max_shift=2, y_only=True)

    parameters = {
        "levels": [4, 2, 1],
        "iterations": 50,
        "minstep": 5e-2,
        "relax": 0.2,
        "recon_algorithm": "SIRT",
        "recon_iterations": 50,
    }

    # Check for CUDA capability
    use_cuda = astra.use_cuda()

    # Calculate the shifts using projection matching
    pm = projmatch.ProjMatch(shifted, nslice=0, cuda=use_cuda, params=parameters)
    pm.calculate_shifts(show_progressbar=False)
    ali = pm.apply_shifts()

    return {
        "known_shifts": shifted.shifts.data[:, 0],
        "calculated_shifts": -pm.total_shifts,
        "aligned": ali,
        "stack": shifted,
    }


@pytest.mark.usefixtures("aligned_simulation")
class TestProjmatchAlignment:
    """Test projection matching alignment."""

    def test_tracking_precision(self, aligned_simulation):
        """Assert that relative frame-to-frame tracking error is sub-pixel."""
        known = aligned_simulation["known_shifts"]
        calculated = aligned_simulation["calculated_shifts"]

        delta_known = np.diff(known)
        delta_calculated = np.diff(calculated)

        rmse_deltas = np.sqrt(np.mean((delta_calculated - delta_known) ** 2))

        assert rmse_deltas < RMSE_THRESH, (
            f"Shifts exceed RMSE threshold: {rmse_deltas:.4f}"
        )

    def test_trend_correlation(self, aligned_simulation):
        """Assert that calculated shifts correlate with known shifts."""
        known = aligned_simulation["known_shifts"]
        calculated = aligned_simulation["calculated_shifts"]

        r_value, _ = pearsonr(known, calculated)
        r_value = cast("float", r_value)
        assert r_value > CORR_COEFF, (
            f"Tracking trend correlation is too low: {r_value:.4f}"
        )

    def test_shift_application(self, aligned_simulation):
        stack = aligned_simulation["stack"]
        ali = aligned_simulation["aligned"]
        known = aligned_simulation["known_shifts"]
        calculated = aligned_simulation["calculated_shifts"]

        assert isinstance(ali, etspy.TomoStack)
        assert ali.axes_manager.signal_shape == stack.axes_manager.signal_shape
        assert ali.axes_manager.navigation_shape == stack.axes_manager.navigation_shape
        np.testing.assert_allclose(
            ali.shifts.data[:, 0],
            known - calculated,
            atol=1e-2,
            err_msg="Shifts in aligned stack do not match expected values",
        )
