"""Test background estimation and its use in alignment and transformation."""

import numpy as np
import pytest

import etspy.api as etspy
from etspy import align
from etspy._background import background_mode, projected_mass


def _disc_stack(background, noise=5.0, shape=(5, 64, 96), seed=0):
    """Projections of a bright disc (about 30% of the area) on a background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[1], : shape[2]]
    disc = 1000.0 * (((yy - shape[1] / 2) ** 2 + (xx - shape[2] / 2) ** 2) < 24**2)
    data = disc + np.asarray(background, dtype=float).reshape(-1, 1, 1)
    data = data + rng.normal(0, noise, shape)
    return etspy.TomoStack(
        data.astype(np.float32),
        tilts=np.linspace(-60, 60, shape[0]),
    )


class TestBackgroundEstimate:
    """Test the histogram-mode background estimate and projected mass."""

    @pytest.mark.parametrize("background", [0.0, 882.0, -31886.0, 4000.0])
    def test_background_mode(self, background):
        stack = _disc_stack(background)
        for image in stack.data:
            assert background_mode(image) == pytest.approx(background, abs=1.0)

    def test_background_mode_large_image(self):
        # large images are subsampled when estimating the background
        image = np.random.default_rng(0).normal(250, 5, (2048, 2048))
        image[500:1500, 500:1500] += 1000
        assert background_mode(image) == pytest.approx(250, abs=1.0)

    def test_projected_mass(self):
        stack = _disc_stack(-31886.0)
        mass = projected_mass(stack.data)
        assert mass.shape == stack.data.shape
        assert mass.min() == 0
        assert np.median(mass[:, :8, :]) < 5  # noqa: PLR2004
        assert np.median(mass[:, 28:36, 44:52]) == pytest.approx(1000, abs=5)


class TestBackgroundFill:
    """Test that uncovered pixels are filled with the background level."""

    @pytest.mark.parametrize("method", ["fourier", "interp"])
    @pytest.mark.parametrize("background", [0.0, -31886.0])
    def test_apply_shifts_fill(self, method, background):
        stack = _disc_stack(background)
        shifts = np.tile([6.0, -4.0], (5, 1))
        shifted = align.apply_shifts(stack, shifts, method=method)
        # rows uncovered by the +6 pixel y-shift and columns by the -4 pixel x-shift
        np.testing.assert_allclose(shifted.data[:, :5, :], background, atol=1.0)
        np.testing.assert_allclose(shifted.data[:, :, -3:], background, atol=1.0)

    def test_apply_shifts_keeps_negative_values(self):
        # negative noise in background-removed data must not be clipped
        stack = _disc_stack(0.0, noise=20.0)
        shifted = align.apply_shifts(stack, np.tile([2.0, 0.0], (5, 1)))
        assert shifted.data.min() < -40  # noqa: PLR2004

    def test_apply_shifts_integer_data(self):
        data = np.full((3, 32, 32), 1000, np.uint16)
        data[:, 10:20, 10:20] = 3000
        stack = etspy.TomoStack(data, tilts=np.array([-10.0, 0.0, 10.0]))
        shifted = align.apply_shifts(stack, np.tile([0.5, 0.0], (3, 1)), "interp")
        assert np.issubdtype(shifted.data.dtype, np.floating)
        # half-pixel shift of an edge gives an intermediate (non-integer) value
        assert 1000 < shifted.data[0, 10, 15] < 3000  # noqa: PLR2004

    @pytest.mark.parametrize("background", [0.0, -31886.0])
    def test_trans_stack_fill(self, background):
        stack = _disc_stack(background)
        transformed = stack.trans_stack(xshift=5, yshift=-5, angle=10)
        corners = transformed.data[:, :3, :3]
        np.testing.assert_allclose(corners, background, atol=1.0)

    def test_trans_stack_integer_nearest(self):
        data = np.full((3, 32, 32), 1000, np.uint16)
        data[:, 10:20, 10:20] = 3000
        stack = etspy.TomoStack(data, tilts=np.array([-10.0, 0.0, 10.0]))
        transformed = stack.trans_stack(xshift=3, interpolation="nearest")
        assert transformed.data.dtype == np.uint16
        np.testing.assert_array_equal(transformed.data[:, :, :3], 1000)
        assert transformed.data.max() == 3000  # noqa: PLR2004
