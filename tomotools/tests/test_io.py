import tomotools.api as tomotools
from tomotools.base import TomoStack
import hyperspy.api as hs
import os
import numpy as np
import pytest

tomotools_path = os.path.dirname(tomotools.__file__)


class TestMRC:

    def test_load_hspy(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        assert type(stack) is TomoStack
        assert stack.axes_manager.signal_shape == (256, 256)
        assert stack.axes_manager.navigation_shape == (77,)
        assert stack.__repr__()[0:10] == '<TomoStack'

    def test_numpy_to_stack(self):
        stack = tomotools.io.numpy_to_tomo_stack(
            np.random.random([50, 100, 100]))
        assert type(stack) is tomotools.TomoStack
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (50,)

    def test_signal_to_stack(self):
        signal = hs.signals.Signal2D(np.random.random([50, 100, 100]))
        stack = tomotools.io.signal_to_tomo_stack(signal)
        assert stack.axes_manager[0].name == 'Tilt'
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (50,)

    def test_signal_to_stack_with_tiltsignal(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        tilt_signal = tomotools.load(filename)
        tilt_signal.set_tilts(-76, 2)
        signal = hs.signals.Signal2D(np.random.random([77, 100, 100]))
        stack = tomotools.io.signal_to_tomo_stack(signal,
                                                  tilt_signal=tilt_signal)
        assert stack.axes_manager[0].name == 'Tilt'
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (77,)
        assert stack.axes_manager[0].axis.all() == \
            tilt_signal.axes_manager[0].axis.all()

    def test_load_dm(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.dm3")
        stack = tomotools.load(filename)
        assert stack.axes_manager[0].name == 'Tilt'
        assert stack.axes_manager[0].offset == -90.0
        assert stack.axes_manager.signal_shape == (64, 64)
        assert stack.axes_manager.navigation_shape == (91,)

    def test_load_unknown(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.NONE")
        with pytest.raises(ValueError):
            tomotools.load(filename)
