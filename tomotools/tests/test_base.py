import os
import matplotlib
import tomotools.api as tomotools
import pytest
import numpy as np
import sys
import io

tomotools_path = os.path.dirname(tomotools.__file__)


class TestTomoStack:

    def test_correlation_check(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        fig = stack.test_correlation()
        assert type(fig) is matplotlib.figure.Figure
        assert len(fig.axes) == 3

    def test_image_filter_median(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        filt = stack.inav[0:10].filter(method='median')
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_image_filter_sobel(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        filt = stack.inav[0:10].filter(method='sobel')
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_image_filter_both(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        filt = stack.inav[0:10].filter(method='both')
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_image_filter_bpf(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        filt = stack.inav[0:10].filter(method='bpf')
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_image_filter_wrong_name(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        with pytest.raises(ValueError):
            stack.inav[0:10].filter(method='WRONG')

    def test_image_filter_none(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        filt = stack.inav[0:10].filter(method=None)
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_stack_normalize(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        norm = stack.normalize()
        assert norm.axes_manager.navigation_shape == \
            stack.axes_manager.navigation_shape
        assert norm.axes_manager.signal_shape == \
            stack.axes_manager.signal_shape
        assert norm.data.min() == 0.0

    def test_stack_invert(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        invert = stack.invert()
        hist, bins = np.histogram(stack.data)
        hist_inv, bins_inv = np.histogram(invert.data)
        assert hist[0] > hist_inv[0]

    def test_stack_stats(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        stdout = sys.stdout
        sys.stdout = io.StringIO()

        stack.stats()

        out = sys.stdout.getvalue()
        sys.stdout = stdout
        out = out.split('\n')

        assert out[0] == 'Mean: %.1f' % stack.data.mean()
        assert out[1] == 'Std: %.2f' % stack.data.std()
        assert out[2] == 'Max: %.1f' % stack.data.max()
        assert out[3] == 'Min: %.1f' % stack.data.min()

    def test_test_align_no_slices(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        stack.test_align()
        fig = matplotlib.pylab.gcf()
        assert len(fig.axes) == 3

    def test_test_align_with_angle(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        stack.test_align(angle=3.0)
        fig = matplotlib.pylab.gcf()
        assert len(fig.axes) == 3

    def test_test_align_with_xshift(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        stack.test_align(xshift=3.0)
        fig = matplotlib.pylab.gcf()
        assert len(fig.axes) == 3

    def test_test_align_with_thickness(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        stack.test_align(thickness=200)
        fig = matplotlib.pylab.gcf()
        assert len(fig.axes) == 3

    def test_set_tilts(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        stack.set_tilts(-50, 5)
        assert stack.axes_manager[0].name == "Tilt"
        assert stack.axes_manager[0].scale == 5
        assert stack.axes_manager[0].units == "degrees"
        assert stack.axes_manager[0].offset == -50
        assert stack.axes_manager[0].axis.all() == \
            np.arange(-50, stack.data.shape[0] * 5 + -50, 5).all()

    def test_sirt_error(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.load(filename)
        # stack.set_tilts(-76, 2)
        rec_stack, error = stack.recon_error(128, iterations=50,
                                             constrain=True, cuda=False)
        print(error.shape)
        print(rec_stack.shape)
        assert error.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == stack.data.shape[1:]
        assert (1 - (3.8709e12 / error[0])) < 0.001
        assert (1 - (2.8624e12 / error[1])) < 0.001
