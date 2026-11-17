"""Test dataset generation features of ETSpy."""

from unittest.mock import patch

import numpy as np
import pytest

from etspy import datasets as ds
from etspy.api import TomoStack


class TestExptlDatasetRetrieval:
    """Test retrieving experimental data."""

    def test_get_experimental_data(self):
        stack = ds.get_needle_data()
        assert isinstance(stack, TomoStack)
        assert np.all(stack.shifts.data == 0)
        assert np.all(stack.tilts.data.T == np.arange(-76, 78, 2))  # noqa: SIM300
        assert stack.axes_manager["Projections"].units == "degrees"  # type: ignore
        assert stack.axes_manager["x"].units == "nm"  # type: ignore
        assert stack.axes_manager["y"].scale == pytest.approx(3.36)  # type: ignore

    def test_get_aligned_experimental_data(self):
        stack = ds.get_needle_data(aligned=True)
        assert isinstance(stack, TomoStack)
        assert np.all(stack.shifts.data[1:, :] != 0)
        assert np.all(stack.tilts.data.T == np.arange(-76, 78, 2))  # noqa: SIM300
        assert stack.axes_manager["Projections"].units == "degrees"  # type: ignore
        assert stack.axes_manager["x"].units == "nm"  # type: ignore
        assert stack.axes_manager["y"].scale == pytest.approx(3.36)  # type: ignore


class TestSimDatasetRetrieval:
    """Test retrieving simulated data."""

    def test_get_simulated_series(self):
        stack = ds.get_catalyst_data()
        assert isinstance(stack, TomoStack)
        assert stack.axes_manager["Projections"].size == 90  # noqa: PLR2004 # type: ignore
        assert stack.axes_manager["x"].units == "nm"  # type: ignore
        assert np.all(stack.tilts.data.T == np.arange(0, 180, 2))  # noqa: SIM300
        assert np.all(stack.shifts.data == 0)

    # The options are tested on a small crop of the series, to avoid loading the
    # full 90 x 600 x 600 file each time.

    @pytest.fixture
    def small_catalyst(self, catalyst_stack):
        """Patch the file read so that get_catalyst_data starts from a small crop."""
        small = catalyst_stack.isig[200:264, 200:264].deepcopy()
        with patch.object(ds.etspy, "load", return_value=small):
            yield small

    def test_get_misaligned_simulated_series(self, small_catalyst):
        stack = ds.get_catalyst_data(misalign=True)
        assert isinstance(stack, TomoStack)
        assert stack.data.shape == small_catalyst.data.shape
        np.testing.assert_array_equal(stack.tilts.data, small_catalyst.tilts.data)
        assert np.all(stack.shifts.data != 0)

    def test_get_noisy_simulated_series(self, small_catalyst):
        stack = ds.get_catalyst_data(noise=True)
        assert isinstance(stack, TomoStack)
        noise = stack.data - small_catalyst.data
        mean = small_catalyst.data.mean()
        # zero-mean noise with a standard deviation of 0.2 x the mean intensity
        assert abs(noise.mean()) < 0.01 * mean
        np.testing.assert_allclose(noise.std(), 0.2 * mean, rtol=0.02)

    def test_get_noisy_misaligned_simulated_series(self, small_catalyst):
        stack = ds.get_catalyst_data(noise=True, misalign=True)
        assert isinstance(stack, TomoStack)
        assert np.all(stack.shifts.data != 0)
        assert not np.allclose(stack.data, small_catalyst.data)
