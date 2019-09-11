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

    def test_tilt_align_com(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        reg = stack.stack_register('PC')
        ali = reg.tilt_align(method='CoM', locs=[64, 128, 192])
        tilt_axis = ali.original_metadata.tiltaxis
        assert 100 * abs((-2.3 - tilt_axis) / -2.3) < 1.0

    def test_tilt_align_maximage(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        reg = stack.stack_register('PC')
        ali = reg.tilt_align(method='MaxImage')
        tilt_axis = ali.original_metadata.tiltaxis
        assert 100 * abs((-2.3 - tilt_axis) / -2.3) < 1.0

    def test_align_to_other(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        stack2 = stack.deepcopy()
        reg = stack.stack_register('PC')
        reg2 = reg.align_other(stack2)
        diff = reg.data - reg2.data
        assert diff.sum() == 0.0
