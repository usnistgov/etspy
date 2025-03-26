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

    def test_blur_convolve_2d(self):
        self.sino = stack.data[:, :, 300]
        blurred = proj_match.blur_convolve(self.sino, 4)
        assert isinstance(blurred, np.ndarray)
        assert blurred.shape == self.sino.shape

    def test_blur_convolve_3d(self):
        self.sino = stack.data[:, :, 300]
        blurred = proj_match.blur_convolve(self.sino, 4)
        assert isinstance(blurred, np.ndarray)
        assert blurred.shape == self.sino.shape

    def test_blur_edges(self):
        self.sino = stack.data[:, :, 300]
        blurred = proj_match.blur_edges(self.sino, 4)
        assert isinstance(blurred, np.ndarray)
        assert blurred.shape == self.sino.shape

    def test_high_pass_filter(self):
        self.sino = stack.data[:, :, 300]
        filtered = proj_match.high_pass_filter(self.sino, 5)
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape == self.sino.shape

    def test_high_pass_fourier_filter(self):
        self.sino = stack.data[:, :, 300]
        filtered = proj_match.high_pass_fourier_filter(self.sino, 0.1)
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape == self.sino.shape

    def test_high_pass_fourier_filter_zero_sigma(self):
        self.sino = stack.data[:, :, 300]
        filtered = proj_match.high_pass_fourier_filter(self.sino, 0.0)
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape == self.sino.shape

    def test_interpolate_ft(self):
        self.sino = stack.data[:, :, 300]
        factor = 4
        blurred = proj_match.blur_convolve(self.sino, factor)
        downsampled = proj_match.interpolate_ft(blurred, factor)
        assert isinstance(downsampled, np.ndarray)
        assert downsampled.shape == (self.sino.shape[0], self.sino.shape[1] // factor)

    def test_shift_sinogram(self):
        self.sino = stack.data[:, :, 300]
        shifts = np.ones(stack.data.shape[0])
        shifted = proj_match.shift_sinogram(self.sino, shifts)
        assert isinstance(shifted, np.ndarray)
        assert shifted.shape == self.sino.shape

    def test_shift_sinogram_zero_shifts(self):
        self.sino = stack.data[:, :, 300]
        shifts = np.zeros(stack.data.shape[0])
        shifted = proj_match.shift_sinogram(self.sino, shifts)
        assert isinstance(shifted, np.ndarray)
        assert shifted.shape == self.sino.shape

    def test_sino_gradient(self):
        self.sino = stack.data[:, :, 300]
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

    def test_shift_calculation_fbp(self):
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
        assert isinstance(pm.total_shifts, np.ndarray)
        assert np.sum(np.abs(pm.total_shifts)) > 0.0
        assert pm.total_shifts.shape[0] == pm.nangles

    def test_shift_calculation_sirt(self):
        sino = stack.isig[300:301, :].rebin(scale=[1, 1, 4])
        params = {
            "levels": [2, 1],
            "iterations": 200,
            "minstep": 1e-2,
            "relax": 0.1,
            "recon_algorithm": "SIRT",
            "recon_iterations": 5,
        }
        pm = projmatch.ProjMatch(sino, cuda=False, params=params)
        pm.calculate_shifts()
        shift_diff = shifts + 4 * pm.total_shifts
        assert np.quantile(shift_diff, 0.80) < 1.0
        assert isinstance(pm.total_shifts, np.ndarray)
        assert np.sum(np.abs(pm.total_shifts)) > 0.0
        assert pm.total_shifts.shape[0] == pm.nangles

    def test_update_geometries(self):
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
        pm.update_geometries(200)
        assert astra.data2d.get(pm.rec_id).shape == (200, 200)


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
        assert isinstance(pm.total_shifts, np.ndarray)
        assert np.sum(np.abs(pm.total_shifts)) > 0.0
        assert pm.total_shifts.shape[0] == pm.nangles
