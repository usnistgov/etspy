from tomotools import datasets as ds
import tomotools.api as tomotools


class TestExptlDatasetRetrieval:

    def test_get_experimental_data(self):
        stack = ds.get_needle_data()
        assert type(stack) is tomotools.TomoStack

    def test_get_aligned_experimental_data(self):
        stack = ds.get_needle_data(aligned=True)
        assert type(stack) is tomotools.TomoStack


class TestSimDatasetRetrieval:
    def test_get_simulated_series(self):
        stack = ds.get_catalyst_tilt_series()
        assert type(stack) is tomotools.TomoStack

    def test_get_misaligned_simulated_series(self, misalign=True):
        stack = ds.get_catalyst_tilt_series(misalign=True)
        assert type(stack) is tomotools.TomoStack

    def test_get_noisy_simulated_series(self, noise=True):
        stack = ds.get_catalyst_tilt_series()
        assert type(stack) is tomotools.TomoStack

    def test_get_noisy_misaligned_simulated_series(self, noise=True,
                                                   misalign=True):
        stack = ds.get_catalyst_tilt_series()
        assert type(stack) is tomotools.TomoStack
