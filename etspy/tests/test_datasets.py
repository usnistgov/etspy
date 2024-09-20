import etspy.api as etspy
from etspy import datasets as ds


class TestExptlDatasetRetrieval:

    def test_get_experimental_data(self):
        stack = ds.get_needle_data()
        assert type(stack) is etspy.TomoStack

    def test_get_aligned_experimental_data(self):
        stack = ds.get_needle_data(aligned=True)
        assert type(stack) is etspy.TomoStack


class TestSimDatasetRetrieval:
    def test_get_simulated_series(self):
        stack = ds.get_catalyst_data()
        assert type(stack) is etspy.TomoStack

    def test_get_misaligned_simulated_series(self, misalign=True):
        stack = ds.get_catalyst_data(misalign=True)
        assert type(stack) is etspy.TomoStack

    def test_get_noisy_simulated_series(self):
        stack = ds.get_catalyst_data(noise=True)
        assert type(stack) is etspy.TomoStack

    def test_get_noisy_misaligned_simulated_series(self, noise=True, misalign=True):
        stack = ds.get_catalyst_data()
        assert type(stack) is etspy.TomoStack
