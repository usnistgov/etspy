"""Test dataset generation features of ETSpy."""

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
        assert stack.axes_manager["Projections"].units == "degrees" # type: ignore
        assert stack.axes_manager["x"].units == "nm" # type: ignore
        assert stack.axes_manager["y"].scale == pytest.approx(3.36) # type: ignore

    def test_get_aligned_experimental_data(self):
        stack = ds.get_needle_data(aligned=True)
        assert isinstance(stack, TomoStack)
        assert np.all(stack.shifts.data[1:,:] != 0)
        assert np.all(stack.tilts.data.T == np.arange(-76, 78, 2))  # noqa: SIM300
        assert stack.axes_manager["Projections"].units == "degrees" # type: ignore
        assert stack.axes_manager["x"].units == "nm" # type: ignore
        assert stack.axes_manager["y"].scale == pytest.approx(3.36) # type: ignore

class TestSimDatasetRetrieval:
    """Test retrieving simulated data."""

    def test_get_simulated_series(self):
        stack = ds.get_catalyst_data()
        assert isinstance(stack, TomoStack)
        assert stack.axes_manager["Projections"].size == 90  # noqa: PLR2004 # type: ignore
        assert stack.axes_manager["x"].units == "nm" # type: ignore
        assert np.all(stack.tilts.data.T == np.arange(0, 180, 2))  # noqa: SIM300
        assert np.all(stack.shifts.data == 0)

    def test_get_misaligned_simulated_series(self):
        stack = ds.get_catalyst_data(misalign=True)
        assert isinstance(stack, TomoStack)
        assert stack.axes_manager["Projections"].size == 90 # type: ignore  # noqa: PLR2004
        assert stack.axes_manager["x"].units == "nm" # type: ignore
        assert np.all(stack.tilts.data.T == np.arange(0, 180, 2))  # noqa: SIM300
        assert np.all(stack.shifts.data != 0)

    def test_get_noisy_simulated_series(self):
        stack = ds.get_catalyst_data(noise=True)
        noiseless_stack = ds.get_catalyst_data(noise=False)
        assert isinstance(stack, TomoStack)
        assert noiseless_stack.data.sum() < stack.data.sum()

    def test_get_noisy_misaligned_simulated_series(self):
        stack = ds.get_catalyst_data(noise=True, misalign=True)
        assert isinstance(stack, TomoStack)
