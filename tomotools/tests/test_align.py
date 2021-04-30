import tomotools.api as tomotools
import os
import pytest

tomotools_path = os.path.dirname(tomotools.__file__)


class TestAlignStackRegister:

    def test_register_ecc(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        reg = stack.inav[0:20].stack_register('ECC')
        assert type(reg) is tomotools.TomoStack
        assert reg.axes_manager.signal_shape == \
            stack.inav[0:20].axes_manager.signal_shape
        assert reg.axes_manager.navigation_shape == \
            stack.inav[0:20].axes_manager.navigation_shape

    def test_register_pc(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        reg = stack.inav[0:20].stack_register('PC')
        assert type(reg) is tomotools.TomoStack
        assert reg.axes_manager.signal_shape == \
            stack.inav[0:20].axes_manager.signal_shape
        assert reg.axes_manager.navigation_shape == \
            stack.inav[0:20].axes_manager.navigation_shape

    def test_register_com(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        reg = stack.inav[0:20].stack_register('COM')
        assert type(reg) is tomotools.TomoStack
        assert reg.axes_manager.signal_shape == \
            stack.inav[0:20].axes_manager.signal_shape
        assert reg.axes_manager.navigation_shape == \
            stack.inav[0:20].axes_manager.navigation_shape

    def test_register_stackreg(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        reg = stack.inav[0:20].stack_register('StackReg')
        assert type(reg) is tomotools.TomoStack
        assert reg.axes_manager.signal_shape == \
            stack.inav[0:20].axes_manager.signal_shape
        assert reg.axes_manager.navigation_shape == \
            stack.inav[0:20].axes_manager.navigation_shape

    def test_register_com_cl(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        reg = stack.inav[0:20].stack_register('COM-CL')
        assert type(reg) is tomotools.TomoStack
        assert reg.axes_manager.signal_shape == \
            stack.inav[0:20].axes_manager.signal_shape
        assert reg.axes_manager.navigation_shape == \
            stack.inav[0:20].axes_manager.navigation_shape

    def test_register_unknown_method(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        with pytest.raises(ValueError):
            stack.inav[0:20].stack_register('WRONG')

    def test_tilt_align_com(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        stack.axes_manager[0].offset = -76
        stack.axes_manager[0].scale = 2
        reg = stack.stack_register('PC')
        ali = reg.tilt_align(method='CoM', locs=[64, 128, 192])
        tilt_axis = ali.metadata.Tomography.tiltaxis
        assert abs(-2.7 - tilt_axis) < 1.0

    def test_tilt_align_maximage(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        stack.axes_manager[0].offset = -76
        stack.axes_manager[0].scale = 2
        reg = stack.stack_register('PC')
        ali = reg.tilt_align(method='MaxImage')
        tilt_axis = ali.metadata.Tomography.tiltaxis
        assert abs(-2.3 - tilt_axis) < 1.0

    def test_tilt_align_unknown_method(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        with pytest.raises(ValueError):
            stack.tilt_align(method='WRONG')

    def test_align_to_other(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        stack2 = stack.deepcopy()
        reg = stack.stack_register('PC')
        reg2 = reg.align_other(stack2)
        diff = reg.data - reg2.data
        assert diff.sum() == 0.0

    def test_align_to_other_no_alignment(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        stack2 = stack.deepcopy()
        reg = stack.deepcopy()
        with pytest.raises(ValueError):
            reg.align_other(stack2)

    def test_align_to_other_with_crop(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        reg = stack.stack_register('PC', crop=True)
        reg2 = reg.align_other(stack)
        diff = reg.data - reg2.data
        assert diff.sum() == 0.0
