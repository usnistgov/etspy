"""Test dataset generation features of ETSpy."""

from etspy import datasets as ds
from etspy.api import TomoStack


class TestExptlDatasetRetrieval:
    """Test retrieving experimental data."""

    def test_get_experimental_data(self):
        stack = ds.get_needle_data()
        assert isinstance(stack, TomoStack)

    def test_get_aligned_experimental_data(self):
        stack = ds.get_needle_data(aligned=True)
        assert isinstance(stack, TomoStack)


class TestSimDatasetRetrieval:
    """Test retrieving simulated data."""

    def test_get_simulated_series(self):
        stack = ds.get_catalyst_data()
        assert isinstance(stack, TomoStack)

    def test_get_misaligned_simulated_series(self):
        stack = ds.get_catalyst_data(misalign=True)
        assert isinstance(stack, TomoStack)

    def test_get_noisy_simulated_series(self):
        stack = ds.get_catalyst_data(noise=True)
        assert isinstance(stack, TomoStack)

    def test_get_noisy_misaligned_simulated_series(self):
        stack = ds.get_catalyst_data(noise=True, misalign=True)
        assert isinstance(stack, TomoStack)
