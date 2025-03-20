"""Tests for the projection matching alignment routine of ETSpy."""

import astra
import numpy as np
import pytest

from etspy import datasets as ds
from etspy import projmatch
from etspy.projmatch import proj_match

stack = ds.get_catalyst_data(misalign=True, minshift=-3, maxshift=3, y_only=True)
shifts = stack.shifts.data[:, 0]


class TestProjMatchHelperFunctions:
    """Test helper functions for ProjMatch."""

    def __init__(self):
        self.sino = stack.data[:, :, 300]

    def test_blur_convolve_2d(self):
        blurred = proj_match.blur_convolve(self.sino, 4)
        assert isinstance(blurred, np.ndarray)
        assert blurred.shape == self.sino.shape

    def test_blur_convolve_3d(self):
        blurred = proj_match.blur_convolve(self.sino, 4)
        assert isinstance(blurred, np.ndarray)
        assert blurred.shape == self.sino.shape

    def test_blur_edges(self):
        blurred = proj_match.blur_edges(self.sino, 4)
        assert isinstance(blurred, np.ndarray)
        assert blurred.shape == self.sino.shape

    def test_high_pass_filter(self):
        filtered = proj_match.high_pass_filter(self.sino, 5)
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape == self.sino.shape

    def test_high_pass_fourier_filter(self):
        filtered = proj_match.high_pass_fourier_filter(self.sino, 0.1)
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape == self.sino.shape

    def test_high_pass_fourier_filter_zero_sigma(self):
        filtered = proj_match.high_pass_fourier_filter(self.sino, 0.0)
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape == self.sino.shape

    def test_interpolate_ft(self):
        factor = 4
        blurred = proj_match.blur_convolve(self.sino, factor)
        downsampled = proj_match.interpolate_ft(blurred, factor)
        assert isinstance(downsampled, np.ndarray)
        assert downsampled.shape == (self.sino.shape[0], self.sino.shape[1] // factor)

    def test_sino_gradient(self):
        grad = proj_match.sino_gradient(self.sino)
        assert isinstance(grad, np.ndarray)
        assert grad.shape == self.sino.shape


class TestProjMatch:
    """Test ProjMatch Class."""

    def test_params_is_none(self):
        sino = stack.isig[300:301, :]
        pm = projmatch.ProjMatch(sino, cuda=False, params=None)
        assert isinstance(pm.params, dict)

    def test_three_dimension_not_implemented(self):
        sino = stack.isig[300:303, :]
        with pytest.raises(
            NotImplementedError,
            match="Alignment of 3D stacks is not yet implemented",
        ):
            projmatch.ProjMatch(sino, cuda=False)

    def test_shift_calculation(self):
        sino = stack.isig[300:301, :].rebin(scale=[1, 1, 4])
        params = {
            "levels": [2, 1],
            "iterations": 200,
            "minstep": 1e-2,
            "relax": 0.1,
            "recon_algorithm": "FBP",
            "recon_iterations": None,
        }
        pm = projmatch.ProjMatch(sino, cuda=False, params=params)
        pm.calculate_shifts()
        shift_diff = shifts + 4 * pm.total_shifts
        assert np.quantile(shift_diff, 0.80) < 1.0
        assert isinstance(pm.total_shifts, np.ndarray)
        assert np.sum(np.abs(pm.total_shifts)) > 0.0
        assert pm.total_shifts.shape[0] == pm.nangles


@pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
class TestProjMatchCUDA:
    """Test CUDA functionality of ProjMatch Class."""

    def test_shift_calculation_cuda(self):
        sino = stack.isig[300:301, :].rebin(scale=[1, 1, 4])
        params = {
            "levels": [2, 1],
            "iterations": 200,
            "minstep": 1e-2,
            "relax": 0.1,
            "recon_algorithm": "FBP",
            "recon_iterations": None,
        }
        pm = projmatch.ProjMatch(sino, cuda=True, params=params)
        pm.calculate_shifts()
        shift_diff = shifts + 4 * pm.total_shifts
        assert np.quantile(shift_diff, 0.80) < 1.0
        assert isinstance(pm.total_shifts, np.ndarray)
        assert np.sum(np.abs(pm.total_shifts)) > 0.0
        assert pm.total_shifts.shape[0] == pm.nangles
