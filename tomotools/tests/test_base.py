import os
import matplotlib
import tomotools.api as tomotools
import pytest

my_path = os.path.dirname(__file__)


class TestTomoStack:

    def test_correlation_check(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        fig = stack.test_correlation()
        assert type(fig) is matplotlib.figure.Figure
        assert len(fig.axes) == 3

    def test_image_filter_median(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        filt = stack.inav[0:10].filter(method='median')
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_image_filter_sobel(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        filt = stack.inav[0:10].filter(method='sobel')
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_image_filter_both(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        filt = stack.inav[0:10].filter(method='both')
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_image_filter_bpf(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        filt = stack.inav[0:10].filter(method='bpf')
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_image_filter_wrong_name(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        with pytest.raises(ValueError):
            stack.inav[0:10].filter(method='WRONG')

    def test_image_filter_none(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        filt = stack.inav[0:10].filter(method=None)
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_stack_normalize(self):
        filename = os.path.join(my_path, "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        norm = stack.normalize()
        assert norm.axes_manager.navigation_shape == \
            stack.axes_manager.navigation_shape
        assert norm.axes_manager.signal_shape == \
            stack.axes_manager.signal_shape
        assert norm.data.min() == 0.0
