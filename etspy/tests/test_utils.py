# ruff: noqa: PLR2004

"""Test utility functions of ETSpy."""

import re

import numpy as np
import pytest

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
def multiframe_small_stack():
    """Load multiframe stack."""
    s = load_serialem_multiframe_data()
    return s.isig[500:524, 500:524]


@pytest.mark.skipif(hspy_mrc_broken is True, reason="Hyperspy MRC reader broken")
class TestMultiframeAverage:
    """Test taking a multiframe average of a stack."""

    @pytest.mark.parametrize("ncpus", [1, 2])
    def test_register_serialem_stack(self, multiframe_small_stack, ncpus):
        stack_avg = utils.register_serialem_stack(multiframe_small_stack, ncpus=ncpus)
        assert isinstance(stack_avg, TomoStack)
        assert stack_avg.data.shape == (3, 24, 24)
        self.check_tilts(multiframe_small_stack, stack_avg)

    @staticmethod
    def check_tilts(frames, averaged):
        """Check that the averaged stack has the tilt of each set of frames."""
        ntilts, nframes = frames.data.shape[:2]
        expected = frames.tilts.data.reshape(ntilts, nframes)[:, 0]
        assert np.any(expected != 0)
        np.testing.assert_allclose(averaged.tilts.data.squeeze(), expected)

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

    @pytest.mark.parametrize(
        ("accuracy", "rtol"), [("low", 0.005), ("medium", 0.005), ("high", 0.001)]
    )
    def test_weight_stack(self, aligned_short_stack, accuracy, rtol):
        reg = utils.weight_stack(aligned_short_stack, accuracy=accuracy)
        assert isinstance(reg, TomoStack)
        # the window only removes mass, and brings each projection's mass down to
        # (about) that of the lightest projection
        assert np.all(reg.data <= aligned_short_stack.data + 1e-6)
        mass = aligned_short_stack.data.sum(axis=(1, 2))
        np.testing.assert_allclose(reg.data.sum(axis=(1, 2)), mass.min(), rtol=rtol)

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
        assert est.shape == (20,)
        # equally sloped angles: unique, within (-90, 90], symmetric about 0
        assert len(np.unique(est)) == 20
        assert est.min() > -90
        assert est.max() == pytest.approx(90)
        np.testing.assert_allclose(
            np.sort(est[est < 90]), -np.sort(est[est < 90])[::-1]
        )

    def test_est_angles_error(self):
        with pytest.raises(ValueError, match="N must be an even number"):
            utils.calc_est_angles(11)

    def test_golden_ratio_angles(self):
        gr = utils.calc_golden_ratio_angles(10, 5)
        assert gr.shape == (5,)
        assert len(np.unique(gr)) == 5
        assert np.all(np.abs(gr) <= 5)

    @pytest.mark.parametrize(
        ("center", "expected"), [(None, (50, 50)), ((30, 60), (60, 30))]
    )
    def test_radial_mask(self, center, expected):
        mask = utils.get_radial_mask((100, 100), center)
        assert mask.shape == (100, 100)
        # center is (x, y); the mask is a disk about it in (row, col)
        assert mask[expected]
        np.testing.assert_allclose(np.argwhere(mask).mean(axis=0), expected, atol=0.5)

    def test_radial_mask_non_square(self):
        # A non-square shape with the default center should still produce a
        # usable mask. Previously the default center placed rows//2 in the x
        # slot, which drove the radius negative and returned an all-False mask.
        mask = utils.get_radial_mask((100, 40))
        assert mask.shape == (100, 40)
        assert mask.any()
        # The masked region should be centered on the array.
        assert mask[50, 20]


class TestWeightingFilter:
    """Test weighting filter."""

    @pytest.mark.parametrize("ny", [63, 64])
    @pytest.mark.parametrize(
        "filter_name", ["ram-lak", "shepp-logan", "hanning", "hann", "cosine"]
    )
    def test_weighting_filter_shape(self, ny, filter_name):
        data = np.random.default_rng(0).random((5, ny, 3), np.float32)
        stack = TomoStack(data, tilts=np.arange(5.0))
        filtered = utils.filter_stack(stack, filter_name=filter_name)
        assert filtered.data.shape == data.shape

    @staticmethod
    def _sinusoid_stack(freq, ny=256):
        """Stack whose projections are a sinusoid along y at freq cycles/pixel."""
        y = np.arange(ny)
        profile = np.cos(2 * np.pi * freq * y)
        data = np.broadcast_to(profile[np.newaxis, :, np.newaxis], (3, ny, 2))
        return TomoStack(data.astype(np.float32), tilts=np.arange(3.0))

    def test_weighting_filter_cutoff_removes_high_frequencies(self):
        stack = self._sinusoid_stack(0.4)
        center = slice(64, 192)  # avoid edge effects from zero padding
        full = utils.filter_stack(stack, filter_name="ram-lak", cutoff=0.5)
        cut = utils.filter_stack(stack, filter_name="ram-lak", cutoff=0.25)
        assert np.abs(full.data[:, center]).max() > 0.5
        assert (
            np.abs(cut.data[:, center]).max()
            < 0.05
            * np.abs(
                full.data[:, center],
            ).max()
        )

    def test_weighting_filter_cutoff_keeps_low_frequencies(self):
        # ram-lak is unchanged below the cutoff, so a low-frequency signal is too
        stack = self._sinusoid_stack(0.1)
        center = slice(64, 192)
        full = utils.filter_stack(stack, filter_name="ram-lak", cutoff=0.5)
        cut = utils.filter_stack(stack, filter_name="ram-lak", cutoff=0.25)
        amplitude = np.abs(full.data[:, center]).max()
        np.testing.assert_allclose(
            cut.data[:, center],
            full.data[:, center],
            atol=0.05 * amplitude,
        )

    @pytest.mark.parametrize("cutoff", [0, -0.1, 0.6])
    def test_weighting_filter_bad_cutoff(self, aligned_short_stack, cutoff):
        with pytest.raises(ValueError, match=r"cutoff must be in the range \(0, 0.5\]"):
            utils.filter_stack(aligned_short_stack, cutoff=cutoff)

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
