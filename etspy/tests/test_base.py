"""Tests for base functions of ETSpy."""

import hyperspy.api as hs
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from etspy import datasets as ds
from etspy.base import CommonStack, RecStack, TomoStack

NUM_AXES_THREE = 3


def _set_tomo_metadata(s):
    tomo_metadata = {
        "cropped": False,
        "shifts": np.zeros([s.data.shape[0], 2]),
        "tiltaxis": 0,
        "tilts": np.zeros(s.data.shape[0]),
        "xshift": 0,
        "yshift": 0,
    }
    s.metadata.add_node("Tomography")
    s.metadata.Tomography.add_dictionary(tomo_metadata)
    return s


class TestTomoStack:
    """Test creation of a TomoStack."""

    def test_tomostack_create(self):
        s = hs.signals.Signal2D(np.random.random([10, 100, 100]))
        stack = TomoStack(s)
        assert isinstance(stack, TomoStack)


class TestFiltering:
    """Test filtering of TomoStack data."""

    def test_correlation_check(self):
        stack = ds.get_needle_data()
        fig = stack.test_correlation()
        assert isinstance(fig, Figure)
        assert len(fig.axes) == NUM_AXES_THREE

    def test_image_filter_median(self):
        stack = ds.get_needle_data()
        filt = stack.inav[0:10].filter(method="median")
        assert (
            filt.axes_manager.navigation_shape
            == stack.inav[0:10].axes_manager.navigation_shape
        )
        assert (
            filt.axes_manager.signal_shape == stack.inav[0:10].axes_manager.signal_shape
        )

    def test_image_filter_sobel(self):
        stack = ds.get_needle_data()
        filt = stack.inav[0:10].filter(method="sobel")
        assert (
            filt.axes_manager.navigation_shape
            == stack.inav[0:10].axes_manager.navigation_shape
        )
        assert (
            filt.axes_manager.signal_shape == stack.inav[0:10].axes_manager.signal_shape
        )

    def test_image_filter_both(self):
        stack = ds.get_needle_data()
        filt = stack.inav[0:10].filter(method="both")
        assert (
            filt.axes_manager.navigation_shape
            == stack.inav[0:10].axes_manager.navigation_shape
        )
        assert (
            filt.axes_manager.signal_shape == stack.inav[0:10].axes_manager.signal_shape
        )

    def test_image_filter_bpf(self):
        stack = ds.get_needle_data()
        filt = stack.inav[0:10].filter(method="bpf")
        assert (
            filt.axes_manager.navigation_shape
            == stack.inav[0:10].axes_manager.navigation_shape
        )
        assert (
            filt.axes_manager.signal_shape == stack.inav[0:10].axes_manager.signal_shape
        )

    def test_image_filter_wrong_name(self):
        stack = ds.get_needle_data()
        bad_name = "WRONG"
        with pytest.raises(
            ValueError,
            match=f"Unknown filter method '{bad_name}'. "
            "Must be 'median', 'sobel', 'both', or 'bpf'",
        ):
            stack.inav[0:10].filter(method="WRONG")


class TestOperations:
    """Test various operations of a TomoStack."""

    def test_stack_normalize(self):
        stack = ds.get_needle_data()
        norm = stack.normalize()
        assert norm.axes_manager.navigation_shape == stack.axes_manager.navigation_shape
        assert norm.axes_manager.signal_shape == stack.axes_manager.signal_shape
        assert norm.data.min() == 0.0

    def test_stack_invert(self):
        im = np.zeros([10, 100, 100])
        im[:, 40:60, 40:60] = 10
        stack = CommonStack(im)
        invert = stack.invert()
        hist, bins = np.histogram(stack.data)
        hist_inv, bins_inv = np.histogram(invert.data)
        assert hist[0] > hist_inv[0]

    def test_stack_stats(self, capsys):
        stack = ds.get_needle_data()
        stack.stats()

        # capture output stream to test print statements
        captured = capsys.readouterr()
        out = captured.out.split("\n")

        assert out[0] == f"Mean: {stack.data.mean():.1f}"
        assert out[1] == f"Std: {stack.data.std():.2f}"
        assert out[2] == f"Max: {stack.data.max():.1f}"
        assert out[3] == f"Min: {stack.data.min():.1f}"

    def test_set_tilts(self):
        stack = ds.get_needle_data()
        start, increment = -50, 5
        stack.set_tilts(start, increment)
        assert stack.axes_manager[0].name == "Tilt"
        assert stack.axes_manager[0].scale == increment
        assert stack.axes_manager[0].units == "degrees"
        assert stack.axes_manager[0].offset == start
        assert (
            stack.axes_manager[0].axis.all()
            == np.arange(
                start,
                stack.data.shape[0] * increment + start,
                increment,
            ).all()
        )

    def test_set_tilts_no_metadata(self):
        stack = ds.get_needle_data()
        del stack.metadata.Tomography
        start, increment = -50, 5
        stack.set_tilts(start, increment)
        assert stack.axes_manager[0].name == "Tilt"
        assert stack.axes_manager[0].scale == increment
        assert stack.axes_manager[0].units == "degrees"
        assert stack.axes_manager[0].offset == start
        assert (
            stack.axes_manager[0].axis.all()
            == np.arange(
                start,
                stack.data.shape[0] * increment + start,
                increment,
            ).all()
        )


class TestTestAlign:
    """Test test alignment of a TomoStack."""

    def test_test_align_no_slices(self):
        stack = ds.get_needle_data(aligned=True)
        stack.test_align()
        fig = plt.gcf()
        assert len(fig.axes) == NUM_AXES_THREE

    def test_test_align_with_angle(self):
        stack = ds.get_needle_data(aligned=True)
        stack.test_align(tilt_rotation=3.0)
        fig = plt.gcf()
        assert len(fig.axes) == NUM_AXES_THREE

    def test_test_align_with_xshift(self):
        stack = ds.get_needle_data(aligned=True)
        stack.test_align(tilt_shift=3.0)
        fig = plt.gcf()
        assert len(fig.axes) == NUM_AXES_THREE

    def test_test_align_with_thickness(self):
        stack = ds.get_needle_data(aligned=True)
        stack.test_align(thickness=200)
        fig = plt.gcf()
        assert len(fig.axes) == NUM_AXES_THREE

    def test_test_align_no_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        stack.test_align(thickness=200, cuda=False)
        fig = plt.gcf()
        assert len(fig.axes) == NUM_AXES_THREE


class TestAlignOther:
    """Test alignment of another TomoStack from an existing one."""

    def test_align_other_no_shifts(self):
        stack = ds.get_needle_data(aligned=False)
        stack2 = stack.deepcopy()
        with pytest.raises(
            ValueError,
            match="No transformations have been applied to this stack",
        ):
            stack.align_other(stack2)

    def test_align_other_with_shifts(self):
        stack = ds.get_needle_data(aligned=True)
        stack2 = stack.deepcopy()
        stack3 = stack.align_other(stack2)
        assert isinstance(stack3, TomoStack)


class TestStackRegister:
    """Test StackReg alignment of a TomoStack."""

    def test_stack_register_unknown_method(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        bad_method = "UNKNOWN"
        with pytest.raises(
            ValueError,
            match=f"Unknown registration method: '{bad_method.lower()}'. "
            "Must be 'PC', 'StackReg', or 'COM'",
        ):
            stack.stack_register(bad_method)

    def test_stack_register_pc(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        stack.metadata.Tomography.shifts = np.zeros([5, 2])
        reg = stack.stack_register("PC")
        assert isinstance(reg, TomoStack)

    def test_stack_register_com(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        stack.metadata.Tomography.shifts = np.zeros([5, 2])
        stack.metadata.Tomography.tilts = stack.metadata.Tomography.tilts[0:5]
        reg = stack.stack_register("COM")
        assert isinstance(reg, TomoStack)

    def test_stack_register_stackreg(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        stack.metadata.Tomography.shifts = np.zeros([5, 2])
        reg = stack.stack_register("COM-CL")
        assert isinstance(reg, TomoStack)

    def test_stack_register_with_crop(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        stack.metadata.Tomography.shifts = np.zeros([5, 2])
        reg = stack.stack_register("PC", crop=True)
        assert isinstance(reg, TomoStack)
        assert np.sum(reg.data.shape) < np.sum(stack.data.shape)


class TestErrorPlots:
    """Test error plots for TomoStack."""

    def test_sirt_error(self):
        stack = ds.get_needle_data(aligned=True)
        rec_stack, error = stack.recon_error(
            128,
            iterations=2,
            constrain=True,
            cuda=False,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (stack.data.shape[1], stack.data.shape[1])

    def test_sirt_error_no_slice(self):
        stack = ds.get_needle_data(aligned=True)
        rec_stack, error = stack.recon_error(
            None,
            iterations=2,
            constrain=True,
            cuda=False,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (stack.data.shape[1], stack.data.shape[1])

    def test_sirt_error_no_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        rec_stack, error = stack.recon_error(
            128,
            iterations=50,
            constrain=True,
            cuda=None,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (stack.data.shape[1], stack.data.shape[1])

    def test_sart_error(self):
        stack = ds.get_needle_data(aligned=True)
        rec_stack, error = stack.recon_error(
            128,
            algorithm="SART",
            iterations=2,
            constrain=True,
            cuda=False,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (stack.data.shape[1], stack.data.shape[1])

    def test_sart_error_no_slice(self):
        stack = ds.get_needle_data(aligned=True)
        rec_stack, error = stack.recon_error(
            None,
            algorithm="SART",
            iterations=2,
            constrain=True,
            cuda=False,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (stack.data.shape[1], stack.data.shape[1])

    def test_sart_error_no_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        rec_stack, error = stack.recon_error(
            128,
            algorithm="SART",
            iterations=50,
            constrain=True,
            cuda=None,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (stack.data.shape[1], stack.data.shape[1])


class TestTiltAlign:
    """Test tilt alignment of a TomoStack."""

    def test_tilt_align_com_axis_zero(self):
        stack = ds.get_needle_data(aligned=True)
        ali = stack.tilt_align("CoM", locs=[64, 100, 114])
        assert isinstance(ali, TomoStack)

    def test_tilt_align_maximage(self):
        stack = ds.get_needle_data(aligned=True)
        stack = stack.inav[0:10]
        ali = stack.tilt_align("MaxImage")
        assert isinstance(ali, TomoStack)

    def test_tilt_align_unknown_method(self):
        stack = ds.get_needle_data(aligned=True)
        bad_method = "UNKNOWN"
        with pytest.raises(
            ValueError,
            match=f"Invalid alignment method: '{bad_method.lower()}'."
            " Must be 'CoM' or 'MaxImage'",
        ):
            stack.tilt_align(bad_method)


class TestTransStack:
    """Test translation of a TomoStack."""

    def test_test_trans_stack_linear(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.trans_stack(1, 1, 1, "linear")
        assert isinstance(shifted, TomoStack)

    def test_test_trans_stack_nearest(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.trans_stack(1, 1, 1, "nearest")
        assert isinstance(shifted, TomoStack)

    def test_test_trans_stack_cubic(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.trans_stack(1, 1, 1, "cubic")
        assert isinstance(shifted, TomoStack)

    def test_test_trans_stack_unknown(self):
        stack = ds.get_needle_data(aligned=True)
        bad_method = "UNKNOWN"
        with pytest.raises(
            ValueError,
            match=f"Interpolation method '{bad_method}' unknown. "
            "Must be 'nearest', 'linear', or 'cubic'",
        ):
            stack.trans_stack(1, 1, 1, bad_method)


class TestReconstruct:
    """Test reconstruction of a TomoStack."""

    def test_cuda_detect(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[:, 120:121].deepcopy()
        rec = slices.reconstruct("FBP", cuda=None)
        assert isinstance(rec, RecStack)


class TestManualAlign:
    """Test manual alignment of a TomoStack."""

    def test_manual_align_positive_x(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, xshift=10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_x(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, xshift=-10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_positive_y(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, yshift=10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_y(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, yshift=-10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_y_positive_x(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, yshift=-10, xshift=10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_x_positive_y(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, yshift=10, xshift=-10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_y_negative_x(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, yshift=-10, xshift=-10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_positive_y_positive_x(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, yshift=10, xshift=10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_no_shifts(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_with_display(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(64, display=True)
        assert isinstance(shifted, TomoStack)


class TestPlotSlices:
    """Test plotting slices of a TomoStack."""

    def test_plot_slices(self):
        rec = RecStack(np.zeros([10, 10, 10]))
        fig = rec.plot_slices()
        assert isinstance(fig, Figure)
