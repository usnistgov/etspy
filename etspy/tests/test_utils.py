"""Test utility functions of ETSpy."""

import re

import numpy as np
import pytest

from etspy import datasets as ds
from etspy import io, utils
from etspy.api import etspy_path
from etspy.base import TomoStack

from . import hspy_mrc_reader_check, load_serialem_multiframe_data

try:
    hspy_mrc_reader_check()
except TypeError:
    hspy_mrc_broken = True
else:
    hspy_mrc_broken = False

@pytest.mark.skipif(hspy_mrc_broken is True, reason="Hyperspy MRC reader broken")
class TestMultiframeAverage:
    """Test taking a multiframe average of a stack."""

    def test_register_serialem_stack(self):
        dirname = etspy_path / "tests" / "test_data" / "SerialEM_Multiframe_Test"
        files = dirname.glob("*.mrc")
        stack = io.load(list(files))
        stack_avg = utils.register_serialem_stack(stack, ncpus=1)
        data_shape = 3
        assert isinstance(stack_avg, TomoStack)
        assert stack_avg.data.shape[0] == data_shape

    def test_register_serialem_stack_multicpu(self):
        dirname = etspy_path / "tests" / "test_data" / "SerialEM_Multiframe_Test"
        files = dirname.glob("*.mrc")
        stack = io.load(list(files))
        stack_avg = utils.register_serialem_stack(stack, ncpus=2)
        data_shape = 3
        assert isinstance(stack_avg, TomoStack)
        assert stack_avg.data.shape[0] == data_shape

    def test_multiaverage(self):
        dirname = etspy_path / "tests" / "test_data" / "SerialEM_Multiframe_Test"
        files = dirname.glob("*.mrc")
        stack = io.load(list(files))
        ntilts, nframes, ny, nx = stack.data.shape
        stack_avg = utils.multiaverage(stack.data[0], nframes, ny, nx)
        assert isinstance(stack_avg, np.ndarray)
        assert stack_avg.shape == (ny, nx)


class TestWeightStack:
    """Test weighting a stack."""

    def test_weight_stack_low(self):
        stack = ds.get_needle_data(aligned=True)
        stack = stack.inav[0:3]
        reg = utils.weight_stack(stack, accuracy="low")
        assert isinstance(reg, TomoStack)

    def test_weight_stack_medium(self):
        stack = ds.get_needle_data(aligned=True)
        stack = stack.inav[0:3]
        reg = utils.weight_stack(stack, accuracy="medium")
        assert isinstance(reg, TomoStack)

    def test_weight_stack_high(self):
        stack = ds.get_needle_data(aligned=True)
        stack = stack.inav[0:3]
        reg = utils.weight_stack(stack, accuracy="high")
        assert isinstance(reg, TomoStack)

    def test_weight_stack_bad_accuracy(self):
        stack = ds.get_needle_data(aligned=True)
        stack = stack.inav[0:3]
        bad_accuracy = "wrong"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid accuracy level "{bad_accuracy}". Must be one of '
                '["low", "medium", or "high"]',
            ),
        ):
            utils.weight_stack(
                stack,
                accuracy="wrong",  # pyright: ignore[reportArgumentType]
            )


class TestHelperUtils:
    """Test helper utilities."""

    def test_est_angles(self):
        est = utils.calc_est_angles(10)
        data_shape = 20
        assert isinstance(est, np.ndarray)
        assert est.shape[0] == data_shape

    def test_est_angles_error(self):
        with pytest.raises(ValueError, match="N must be an even number"):
            utils.calc_est_angles(11)

    def test_golden_ratio_angles(self):
        gr = utils.calc_golden_ratio_angles(10, 5)
        data_shape = 5
        assert isinstance(gr, np.ndarray)
        assert gr.shape[0] == data_shape

    def test_radial_mask_no_center(self):
        mask = utils.get_radial_mask((100, 100), None)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (100, 100)

    def test_radial_mask_with_center(self):
        mask = utils.get_radial_mask((100, 100), (50, 50))
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (100, 100)


class TestWeightingFilter:
    """Test weighting filter."""

    def test_weighting_filter_shepp_logan(self):
        stack = ds.get_needle_data(aligned=True)
        stack = stack.inav[0:3]
        filtered = utils.filter_stack(stack, filter_name="shepp-logan", cutoff=0.5)
        assert isinstance(filtered, TomoStack)

    def test_weighting_filter_ram_lak(self):
        stack = ds.get_needle_data(aligned=True)
        stack = stack.inav[0:3]
        filtered = utils.filter_stack(stack, filter_name="ram-lak", cutoff=0.5)
        assert isinstance(filtered, TomoStack)

    def test_weighting_filter_cosine(self):
        stack = ds.get_needle_data(aligned=True)
        stack = stack.inav[0:3]
        filtered = utils.filter_stack(stack, filter_name="cosine", cutoff=0.5)
        assert isinstance(filtered, TomoStack)

    def test_weighting_filter_shepp_hanning(self):
        stack = ds.get_needle_data(aligned=True)
        stack = stack.inav[0:3]
        filtered = utils.filter_stack(stack, filter_name="hanning", cutoff=0.5)
        assert isinstance(filtered, TomoStack)

    def test_weighting_filter_two_dimensional_data(self):
        stack = ds.get_needle_data(aligned=True)
        stack = stack.inav[0]
        filtered = utils.filter_stack(stack, filter_name="hanning", cutoff=0.5)
        assert isinstance(filtered, TomoStack)

    def test_weighting_filter_bad_filter(self):
        stack = ds.get_needle_data(aligned=True)
        stack = stack.inav[0:3]
        bad_filter = "wrong"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid filter type "{bad_filter}". Must be one of '
                '["ram-lak", "shepp-logan", "hanning", "hann", "cosine", or "cos"]',
            ),
        ):
            utils.filter_stack(
                stack,
                filter_name="wrong",  # pyright: ignore[reportArgumentType]
                cutoff=0.5,
            )

    def test_weighting_filter_bad_stack_shape(self):
        stack = load_serialem_multiframe_data()
        with pytest.raises(
            ValueError,
            match="Method can only be applied to 2 or 3-dimensional stacks",
        ):
            utils.filter_stack(stack)
