import tomotools.api as tomotools
import os

my_path = os.path.dirname(__file__)


class TestAlignStackRegister:

    def test_register_ecc(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        reg = stack.inav[0:20].stack_register('ECC')
        assert type(reg) is tomotools.TomoStack
        assert reg.axes_manager.signal_shape == \
            stack.inav[0:20].axes_manager.signal_shape
        assert reg.axes_manager.navigation_shape == \
            stack.inav[0:20].axes_manager.navigation_shape

    def test_register_pc(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        reg = stack.inav[0:20].stack_register('PC')
        assert type(reg) is tomotools.TomoStack
        assert reg.axes_manager.signal_shape == \
            stack.inav[0:20].axes_manager.signal_shape
        assert reg.axes_manager.navigation_shape == \
            stack.inav[0:20].axes_manager.navigation_shape
