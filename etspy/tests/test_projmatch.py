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


@pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
class TestProjMatchCUDA:
    """Test ProjMatchClass."""

    def test_three_dimension_not_implemented(self):
        sino = stack.isig[300:303, :]
        with pytest.raises(
            NotImplementedError,
            match="Alignment of 3D stacks is not yet implemented",
        ):
            projmatch.ProjMatch(sino, cuda=True)
