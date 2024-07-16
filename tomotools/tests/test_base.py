import matplotlib
from tomotools import datasets as ds
import pytest
import numpy as np
import sys
import io
from tomotools.base import CommonStack, TomoStack, RecStack
import hyperspy.api as hs
# from hyperspy.signals import Signal2D


def _set_tomo_metadata(s):
    tomo_metadata = {"cropped": False,
                     "shifts": np.zeros([s.data.shape[0], 2]),
                     "tiltaxis": 0,
                     "tilts": np.zeros(s.data.shape[0]),
                     "xshift": 0,
                     "yshift": 0}
    s.metadata.add_node("Tomography")
    s.metadata.Tomography.add_dictionary(tomo_metadata)
    return s


class TestTomoStack:
    def test_tomostack_create(self):
        s = hs.signals.Signal2D(np.random.random([10, 100, 100]))
        stack = TomoStack(s)
        assert type(stack) is TomoStack


class TestFiltering:

    def test_correlation_check(self):
        stack = ds.get_needle_data()
        fig = stack.test_correlation()
        assert type(fig) is matplotlib.figure.Figure
        assert len(fig.axes) == 3

    def test_image_filter_median(self):
        stack = ds.get_needle_data()
        filt = stack.inav[0:10].filter(method='median')
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_image_filter_sobel(self):
        stack = ds.get_needle_data()
        filt = stack.inav[0:10].filter(method='sobel')
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_image_filter_both(self):
        stack = ds.get_needle_data()
        filt = stack.inav[0:10].filter(method='both')
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_image_filter_bpf(self):
        stack = ds.get_needle_data()
        filt = stack.inav[0:10].filter(method='bpf')
        assert filt.axes_manager.navigation_shape == \
            stack.inav[0:10].axes_manager.navigation_shape
        assert filt.axes_manager.signal_shape == \
            stack.inav[0:10].axes_manager.signal_shape

    def test_image_filter_wrong_name(self):
        stack = ds.get_needle_data()
        with pytest.raises(ValueError):
            stack.inav[0:10].filter(method='WRONG')


class TestOperations:

    def test_stack_normalize(self):
        stack = ds.get_needle_data()
        norm = stack.normalize()
        assert norm.axes_manager.navigation_shape == \
            stack.axes_manager.navigation_shape
        assert norm.axes_manager.signal_shape == \
            stack.axes_manager.signal_shape
        assert norm.data.min() == 0.0

    def test_stack_invert(self):
        im = np.zeros([10, 100, 100])
        im[:, 40:60, 40:60] = 10
        stack = CommonStack(im)
        invert = stack.invert()
        hist, bins = np.histogram(stack.data)
        hist_inv, bins_inv = np.histogram(invert.data)
        assert hist[0] > hist_inv[0]

    def test_stack_stats(self):
        stack = ds.get_needle_data()
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

    def test_set_tilts(self):
        stack = ds.get_needle_data()
        stack.set_tilts(-50, 5)
        assert stack.axes_manager[0].name == "Tilt"
        assert stack.axes_manager[0].scale == 5
        assert stack.axes_manager[0].units == "degrees"
        assert stack.axes_manager[0].offset == -50
        assert stack.axes_manager[0].axis.all() == \
            np.arange(-50, stack.data.shape[0] * 5 + -50, 5).all()

    def test_set_tilts_no_metadata(self):
        stack = ds.get_needle_data()
        del stack.metadata.Tomography
        stack.set_tilts(-50, 5)
        assert stack.axes_manager[0].name == "Tilt"
        assert stack.axes_manager[0].scale == 5
        assert stack.axes_manager[0].units == "degrees"
        assert stack.axes_manager[0].offset == -50
        assert stack.axes_manager[0].axis.all() == \
            np.arange(-50, stack.data.shape[0] * 5 + -50, 5).all()


class TestTestAlign:

    def test_test_align_no_slices(self):
        stack = ds.get_needle_data(True)
        stack.test_align()
        fig = matplotlib.pylab.gcf()
        assert len(fig.axes) == 3

    def test_test_align_with_angle(self):
        stack = ds.get_needle_data(True)
        stack.test_align(tilt_rotation=3.0)
        fig = matplotlib.pylab.gcf()
        assert len(fig.axes) == 3

    def test_test_align_with_xshift(self):
        stack = ds.get_needle_data(True)
        stack.test_align(tilt_shift=3.0)
        fig = matplotlib.pylab.gcf()
        assert len(fig.axes) == 3

    def test_test_align_with_thickness(self):
        stack = ds.get_needle_data(True)
        stack.test_align(thickness=200)
        fig = matplotlib.pylab.gcf()
        assert len(fig.axes) == 3

    def test_test_align_no_cuda(self):
        stack = ds.get_needle_data(True)
        stack.test_align(thickness=200, cuda=False)
        fig = matplotlib.pylab.gcf()
        assert len(fig.axes) == 3


class TestAlignOther:

    def test_align_other_no_shifts(self):
        stack = ds.get_needle_data(False)
        stack2 = stack.deepcopy()
        with pytest.raises(ValueError):
            stack.align_other(stack2)

    def test_align_other_with_shifts(self):
        stack = ds.get_needle_data(True)
        stack2 = stack.deepcopy()
        stack3 = stack.align_other(stack2)
        assert type(stack3) is TomoStack


class TestStackRegister:

    def test_stack_register_unknown_method(self):
        stack = ds.get_needle_data(False).inav[0:5]
        with pytest.raises(ValueError):
            stack.stack_register('UNKNOWN')

    def test_stack_register_pc(self):
        stack = ds.get_needle_data(False).inav[0:5]
        stack.metadata.Tomography.shifts = np.zeros([5, 2])
        reg = stack.stack_register('PC')
        assert type(reg) is TomoStack

    def test_stack_register_com(self):
        stack = ds.get_needle_data(False).inav[0:5]
        stack.metadata.Tomography.shifts = np.zeros([5, 2])
        stack.metadata.Tomography.tilts = stack.metadata.Tomography.tilts[0:5]
        reg = stack.stack_register('COM')
        assert type(reg) is TomoStack

    def test_stack_register_stackreg(self):
        stack = ds.get_needle_data(False).inav[0:5]
        stack.metadata.Tomography.shifts = np.zeros([5, 2])
        reg = stack.stack_register('COM-CL')
        assert type(reg) is TomoStack

    def test_stack_register_with_crop(self):
        stack = ds.get_needle_data(False).inav[0:5]
        stack.metadata.Tomography.shifts = np.zeros([5, 2])
        reg = stack.stack_register('PC', crop=True)
        assert type(reg) is TomoStack
        assert np.sum(reg.data.shape) < np.sum(stack.data.shape)


class TestErrorPlots:

    def test_sirt_error(self):
        stack = ds.get_needle_data(True)
        rec_stack, error = stack.recon_error(128, iterations=2,
                                             constrain=True, cuda=False)
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] ==\
            (stack.data.shape[1], stack.data.shape[1])

    def test_sirt_error_no_slice(self):
        stack = ds.get_needle_data(True)
        rec_stack, error = stack.recon_error(None, iterations=2,
                                             constrain=True, cuda=False)
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] ==\
            (stack.data.shape[1], stack.data.shape[1])
        assert (1 - (3.8709e12 / error.data[0])) < 0.001
        assert (1 - (2.8624e12 / error.data[1])) < 0.001

    def test_sirt_error_no_cuda(self):
        stack = ds.get_needle_data(True)
        rec_stack, error = stack.recon_error(128, iterations=50,
                                             constrain=True, cuda=None)
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] ==\
            (stack.data.shape[1], stack.data.shape[1])
        assert (1 - (3.8709e12 / error.data[0])) < 0.001
        assert (1 - (2.8624e12 / error.data[1])) < 0.001

    def test_sart_error(self):
        stack = ds.get_needle_data(True)
        rec_stack, error = stack.recon_error(128, algorithm='SART', iterations=2,
                                             constrain=True, cuda=False)
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] ==\
            (stack.data.shape[1], stack.data.shape[1])

    def test_sart_error_no_slice(self):
        stack = ds.get_needle_data(True)
        rec_stack, error = stack.recon_error(None, algorithm='SART', iterations=2,
                                             constrain=True, cuda=False)
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] ==\
            (stack.data.shape[1], stack.data.shape[1])
        assert (1 - (3.8709e12 / error.data[0])) < 0.001
        assert (1 - (2.8624e12 / error.data[1])) < 0.001

    def test_sart_error_no_cuda(self):
        stack = ds.get_needle_data(True)
        rec_stack, error = stack.recon_error(128, algorithm='SART', iterations=50,
                                             constrain=True, cuda=None)
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] ==\
            (stack.data.shape[1], stack.data.shape[1])
        assert (1 - (3.8709e12 / error.data[0])) < 0.001
        assert (1 - (2.8624e12 / error.data[1])) < 0.001


class TestTiltAlign:

    def test_tilt_align_com_axis_zero(self):
        stack = ds.get_needle_data(True)
        ali = stack.tilt_align('CoM', locs=[64, 100, 114])
        assert type(ali) is TomoStack

    def test_tilt_align_maximage(self):
        stack = ds.get_needle_data(True)
        stack = stack.inav[0:10]
        ali = stack.tilt_align('MaxImage')
        assert type(ali) is TomoStack

    def test_tilt_align_unknown_method(self):
        stack = ds.get_needle_data(True)
        with pytest.raises(ValueError):
            stack.tilt_align('UNKNOWN')


class TestTransStack:

    def test_test_trans_stack_linear(self):
        stack = ds.get_needle_data(True)
        shifted = stack.trans_stack(1, 1, 1, 'linear')
        assert type(shifted) is TomoStack

    def test_test_trans_stack_nearest(self):
        stack = ds.get_needle_data(True)
        shifted = stack.trans_stack(1, 1, 1, 'nearest')
        assert type(shifted) is TomoStack

    def test_test_trans_stack_cubic(self):
        stack = ds.get_needle_data(True)
        shifted = stack.trans_stack(1, 1, 1, 'cubic')
        assert type(shifted) is TomoStack

    def test_test_trans_stack_unknown(self):
        stack = ds.get_needle_data(True)
        with pytest.raises(ValueError):
            stack.trans_stack(1, 1, 1, 'UNKNOWN')


class TestReconstruct:
    def test_cuda_detect(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[:, 120:121].deepcopy()
        rec = slices.reconstruct('FBP', cuda=None)
        assert type(rec) is RecStack


class TestManualAlign:
    def test_manual_align_positive_x(self):
        stack = ds.get_needle_data(True)
        shifted = stack.manual_align(128, xshift=10)
        assert type(shifted) is TomoStack

    def test_manual_align_negative_x(self):
        stack = ds.get_needle_data(True)
        shifted = stack.manual_align(128, xshift=-10)
        assert type(shifted) is TomoStack

    def test_manual_align_positive_y(self):
        stack = ds.get_needle_data(True)
        shifted = stack.manual_align(128, yshift=10)
        assert type(shifted) is TomoStack

    def test_manual_align_negative_y(self):
        stack = ds.get_needle_data(True)
        shifted = stack.manual_align(128, yshift=-10)
        assert type(shifted) is TomoStack

    def test_manual_align_negative_y_positive_x(self):
        stack = ds.get_needle_data(True)
        shifted = stack.manual_align(128, yshift=-10, xshift=10)
        assert type(shifted) is TomoStack

    def test_manual_align_negative_x_positive_y(self):
        stack = ds.get_needle_data(True)
        shifted = stack.manual_align(128, yshift=10, xshift=-10)
        assert type(shifted) is TomoStack

    def test_manual_align_negative_y_negative_x(self):
        stack = ds.get_needle_data(True)
        shifted = stack.manual_align(128, yshift=-10, xshift=-10)
        assert type(shifted) is TomoStack

    def test_manual_align_positive_y_positive_x(self):
        stack = ds.get_needle_data(True)
        shifted = stack.manual_align(128, yshift=10, xshift=10)
        assert type(shifted) is TomoStack

    def test_manual_align_no_shifts(self):
        stack = ds.get_needle_data(True)
        shifted = stack.manual_align(128)
        assert type(shifted) is TomoStack

    def test_manual_align_with_display(self):
        stack = ds.get_needle_data(True)
        shifted = stack.manual_align(64, display=True)
        assert type(shifted) is TomoStack


class TestPlotSlices:
    def test_plot_slices(self):
        rec = RecStack(np.zeros([10, 10, 10]))
        fig = rec.plot_slices()
        assert type(fig) is matplotlib.figure.Figure
