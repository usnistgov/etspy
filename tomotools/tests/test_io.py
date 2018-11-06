import tomotools
import os
import numpy as np

my_path = os.path.dirname(__file__)


class TestMRC:

    def test_load_hspy(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        assert type(stack) is tomotools.base.TomoStack
        assert stack.axes_manager.signal_shape == (256, 256)
        assert stack.axes_manager.navigation_shape == (77,)

    def test_numpy_to_stack(self):
        stack = tomotools.io.numpy_to_tomo_stack(np.random.random([50, 100, 100]))
        assert type(stack) is tomotools.base.TomoStack
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (50,)
