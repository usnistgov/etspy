import tomotools
import os
import numpy as np

my_path = os.path.dirname(__file__)


class TestMRC:

    def test_load_hspy(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)

    def test_numpy_to_stack(self):
        stack = tomotools.io.numpy_to_tomo_stack(np.random.random([79, 100, 100]))
