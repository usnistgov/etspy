"""Test utility functions of ETSpy."""

import re

import numpy as np
import pytest

from etspy import datasets as ds
from etspy import utils
from etspy.base import TomoStack

from . import hspy_mrc_reader_check, load_serialem_multiframe_data

try:
    hspy_mrc_reader_check()
except TypeError:
    hspy_mrc_broken = True
else:
    hspy_mrc_broken = False


@pytest.fixture(scope="module")
def aligned_short_stack():
    """Create truncated and spatially registered stack from test data."""
    s = ds.get_needle_data().inav[0:5]
    s = s.stack_register("PC")
    return s


@pytest.fixture(scope="module")
def multiframe_small_stack():
    """Load multiframe stack."""
    s = load_serialem_multiframe_data()
    return s.isig[500:524, 500:524]


@pytest.mark.skipif(hspy_mrc_broken is True, reason="Hyperspy MRC reader broken")
class TestMultiframeAverage:
    """Test taking a multiframe average of a stack."""

    def test_register_serialem_stack(self, multiframe_small_stack):
        stack_avg = utils.register_serialem_stack(
            multiframe_small_stack,
            ncpus=1,
        )
        data_shape = 3
        assert isinstance(stack_avg, TomoStack)
        assert stack_avg.data.shape[0] == data_shape

    def test_register_serialem_stack_multicpu(self, multiframe_small_stack):
        stack_avg = utils.register_serialem_stack(multiframe_small_stack, ncpus=2)
        data_shape = 3
        assert isinstance(stack_avg, TomoStack)
        assert stack_avg.data.shape[0] == data_shape

    def test_multiaverage(self, multiframe_small_stack):
        _, nframes, ny, nx = multiframe_small_stack.data.shape
        stack_avg = utils.multiaverage(
            multiframe_small_stack.data[0],
            nframes,
            ny,
            nx,
        )
        assert isinstance(stack_avg, np.ndarray)
        assert stack_avg.shape == (ny, nx)


class TestWeightStack:
    """Test weighting a stack."""

    def test_weight_stack_low(self, aligned_short_stack):
        reg = utils.weight_stack(aligned_short_stack, accuracy="low")
        assert isinstance(reg, TomoStack)

    def test_weight_stack_medium(self, aligned_short_stack):
        reg = utils.weight_stack(aligned_short_stack, accuracy="medium")
        assert isinstance(reg, TomoStack)

    def test_weight_stack_high(self, aligned_short_stack):
        reg = utils.weight_stack(aligned_short_stack, accuracy="high")
        assert isinstance(reg, TomoStack)

    def test_weight_stack_bad_accuracy(self, aligned_short_stack):
        bad_accuracy = "wrong"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid accuracy level "{bad_accuracy}". Must be one of '
                '["low", "medium", or "high"]',
            ),
        ):
            utils.weight_stack(
                aligned_short_stack,
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

    def test_weighting_filter_shepp_logan(self, aligned_short_stack):
        filtered = utils.filter_stack(
            aligned_short_stack,
            filter_name="shepp-logan",
            cutoff=0.5,
        )
        assert isinstance(filtered, TomoStack)

    def test_weighting_filter_ram_lak(self, aligned_short_stack):
        filtered = utils.filter_stack(
            aligned_short_stack,
            filter_name="ram-lak",
            cutoff=0.5,
        )
        assert isinstance(filtered, TomoStack)

    def test_weighting_filter_cosine(self, aligned_short_stack):
        filtered = utils.filter_stack(
            aligned_short_stack,
            filter_name="cosine",
            cutoff=0.5,
        )
        assert isinstance(filtered, TomoStack)

    def test_weighting_filter_shepp_hanning(self, aligned_short_stack):
        filtered = utils.filter_stack(
            aligned_short_stack,
            filter_name="hanning",
            cutoff=0.5,
        )
        assert isinstance(filtered, TomoStack)

    def test_weighting_filter_two_dimensional_data(self, aligned_short_stack):
        filtered = utils.filter_stack(
            aligned_short_stack,
            filter_name="hanning",
            cutoff=0.5,
        )
        assert isinstance(filtered, TomoStack)

    def test_weighting_filter_bad_filter(self, aligned_short_stack):
        bad_filter = "wrong"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid filter type "{bad_filter}". Must be one of '
                '["ram-lak", "shepp-logan", "hanning", "hann", "cosine", or "cos"]',
            ),
        ):
            utils.filter_stack(
                aligned_short_stack,
                filter_name="wrong",  # pyright: ignore[reportArgumentType]
                cutoff=0.5,
            )

    def test_weighting_filter_bad_stack_shape(self, multiframe_small_stack):
        with pytest.raises(
            ValueError,
            match="Method can only be applied to 2 or 3-dimensional stacks",
        ):
            utils.filter_stack(multiframe_small_stack)
