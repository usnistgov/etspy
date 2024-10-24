"""Tests for base functions of ETSpy."""

import logging
import re
from typing import cast

import hyperspy.api as hs
import numpy as np
import pytest
from hyperspy.axes import UniformDataAxis as Uda
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from etspy import datasets as ds
from etspy.base import RecStack, TomoShifts, TomoStack, TomoTilts

from . import load_serialem_multiframe_data

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
        assert hasattr(stack, "tilts")
        assert hasattr(stack, "shifts")

    def test_remove_projections(self):
        s = ds.get_needle_data(aligned=True)
        s_new = s.remove_projections([0, 5, 10, 15, 20])
        # original shape is (77, 256, 256)
        assert s_new.data.shape == (72, 256, 256)
        assert s_new.tilts.data.shape == (72, 1)
        assert s_new.shifts.data.shape == (72, 2)
        for t in [-76, -66, -56, -46, -36]:
            assert t not in s_new.tilts.data

    def test_remove_projections_none(self):
        s = ds.get_needle_data(aligned=True)
        with pytest.raises(ValueError, match="No projections provided"):
            s.remove_projections(None)

    def test_slicing(self):
        s = ds.get_needle_data()
        s2 = s.inav[:5]
        assert s2.data.shape == (5, 256, 256)
        assert s2.shifts.data.shape == (5, 2)
        assert s2.tilts.data.shape == (5, 1)

    def test_property_deleters(self):
        s = ds.get_needle_data()

        # tilts
        assert np.all(s.tilts.data.squeeze() == np.arange(-76, 78, 2))
        del s.tilts
        assert np.all(s.tilts.data.squeeze() == np.zeros((77, 1)))

        # shifts
        s.shifts = np.random.rand(77,2) + 2  # offset to ensure non-zero
        assert np.all(s.shifts.data != np.zeros((77, 2)))
        del s.shifts
        assert np.all(s.shifts.data == np.zeros((77, 2)))

class TestSlicers:
    """Test inav/isig slicers."""

    def test_tilt_nav_slicer(self):
        stack = ds.get_needle_data()
        t = stack.tilts.inav[:5]
        assert isinstance(t, TomoTilts)
        assert t.data.shape == (5, 1)

    def test_tilt_sig_slicer(self, caplog):
        stack = ds.get_needle_data()
        with caplog.at_level(logging.WARNING):
            t = stack.tilts.isig[:5]
            assert "TomoTilts does not support 'isig' slicing" in caplog.text
        assert isinstance(t, TomoTilts)
        assert t.data.shape == (77, 1)

    def test_shift_nav_slicer(self):
        stack = ds.get_needle_data()
        t = stack.shifts.inav[:5]
        assert isinstance(t, TomoShifts)
        assert t.data.shape == (5, 2)

    def test_shift_sig_slicer(self, caplog):
        stack = ds.get_needle_data()
        with caplog.at_level(logging.WARNING):
            t = stack.shifts.isig[:5]
            # warning should be triggered when slicing the TomoTilts directly
            assert "TomoShifts does not support 'isig' slicing" in caplog.text
        assert isinstance(t, TomoShifts)
        assert t.data.shape == (77, 2)

    def test_tomostack_nav_slicer(self):
        stack = ds.get_needle_data()
        t = stack.inav[:5]
        assert t.data.shape == (5, 256, 256)
        assert t.tilts.data.shape == (5, 1)
        assert t.shifts.data.shape == (5, 2)
        assert isinstance(t, TomoStack)
        assert isinstance(t.tilts, TomoTilts)
        assert isinstance(t.shifts, TomoShifts)

    def test_tomostack_sig_slicer(self, caplog):
        stack = ds.get_needle_data()
        with caplog.at_level(logging.WARNING):
            t = stack.isig[:5, :10]
            # warning should not be triggered when slicing the TomoStack
            assert "TomoShifts does not support 'isig' slicing" not in caplog.text
        assert t.data.shape == (77, 10, 5)
        assert t.tilts.data.shape == (77, 1)
        assert t.shifts.data.shape == (77, 2)
        assert isinstance(t, TomoStack)
        assert isinstance(t.tilts, TomoTilts)
        assert isinstance(t.shifts, TomoShifts)

    def test_two_d_tomo_stack_slicing(self):
        stack = load_serialem_multiframe_data()
        assert stack.axes_manager.shape == (2, 3, 1024, 1024)
        assert stack.data.shape == (3, 2, 1024, 1024)
        assert stack.axes_manager[0].name == "Frames"       # type: ignore
        assert stack.axes_manager[0].units == "images"      # type: ignore
        assert stack.axes_manager[1].name == "Projections"  # type: ignore
        assert stack.axes_manager[1].units == "degrees"     # type: ignore
        assert stack.axes_manager[2].name == "x"            # type: ignore
        assert stack.axes_manager[2].units == "nm"          # type: ignore
        assert stack.axes_manager[3].name == "y"            # type: ignore
        assert stack.axes_manager[3].units == "nm"          # type: ignore

        # test inav and isig together with ranges
        t = stack.inav[:1, :2].isig[:20, :120]
        assert t.axes_manager.shape == (1, 2, 20, 120)
        assert t.data.shape == (2, 1, 120, 20)
        assert t.tilts.axes_manager.shape == (1, 2, 1)
        assert t.shifts.axes_manager.shape == (1, 2, 2)

        # test extracting single projection
        t2 = stack.isig[:20, :120].inav[:, 2]
        assert t2.axes_manager.shape == (2, 20, 120)
        assert t2.data.shape == (2, 120, 20)
        assert t2.axes_manager[0].name == "Frames" # type: ignore
        assert t2.tilts.axes_manager.navigation_shape == (2,)
        assert t2.shifts.axes_manager.navigation_shape == (2,)

        # test extracting single frame
        t3 = stack.isig[:20, :120].inav[1, :]
        assert t3.axes_manager.shape == (3, 20, 120)
        assert t3.data.shape == (3, 120, 20)
        assert t3.axes_manager[0].name == "Projections" # type: ignore
        assert t3.tilts.axes_manager.navigation_shape == (3,)
        assert t3.shifts.axes_manager.navigation_shape == (3,)

        # test extracting single frame and projection
        t4 = stack.isig[:20, :120].inav[1, 1]
        assert t4.axes_manager.shape == (20, 120)
        assert t4.data.shape == (120, 20)
        assert t4.axes_manager[0].name == "x" # type: ignore
        assert t4.axes_manager[1].name == "y" # type: ignore
        assert t4.tilts.axes_manager.navigation_shape == ()
        assert t4.shifts.axes_manager.navigation_shape == ()
        assert t4.tilts.data[0] == pytest.approx(-0.000488)

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
            match=re.escape(
                f'Invalid filter method "{bad_name}". '
                'Must be one of ["median", "bpf", "both", or "sobel"]',
            ),
        ):
            stack.inav[0:10].filter(method="WRONG") # type: ignore


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
        stack = TomoStack(im)
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
        ax = cast(Uda, stack.axes_manager[0])
        assert ax.name == "Projections"
        assert ax.scale == increment
        assert ax.units == "degrees"
        assert ax.offset == start
        assert (
            ax.axis.all()
            == np.arange(
                start,
                stack.data.shape[0] * increment + start,
                increment,
            ).all()
        )

    def test_set_tilts_no_metadata(self):
        stack = ds.get_needle_data()
        del stack.metadata.Tomography  # pyright: ignore[reportAttributeAccessIssue]
        start, increment = -50, 5
        stack.set_tilts(start, increment)
        ax = cast(Uda, stack.axes_manager[0])
        assert ax.name == "Projections"
        assert ax.scale == increment
        assert ax.units == "degrees"
        assert ax.offset == start
        assert (
            ax.axis.all()
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
        assert (
            stack.metadata.Tomography.xshift == stack2.metadata.Tomography.xshift # type: ignore
        )
        assert (
            stack3.metadata.Tomography.xshift == 2 * stack2.metadata.Tomography.xshift # type: ignore
        )


class TestStackRegister:
    """Test StackReg alignment of a TomoStack."""

    def test_stack_register_unknown_method(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        bad_method = "UNKNOWN"
        with pytest.raises(
            TypeError,
            match=re.escape(
                f'Invalid registration method "{bad_method}". '
                'Must be one of ["StackReg", "PC", "COM", or "COM-CL"].',
            ),
        ):
            stack.stack_register(bad_method) # type: ignore

    def test_stack_register_pc(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        reg = stack.stack_register("PC")
        assert isinstance(reg, TomoStack)

    def test_stack_register_com(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        reg = stack.stack_register("COM")
        assert isinstance(reg, TomoStack)

    def test_stack_register_stackreg(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        reg = stack.stack_register("COM-CL")
        assert isinstance(reg, TomoStack)

    def test_stack_register_with_crop(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
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
        ali = stack.tilt_align("CoM", slices=np.array([64, 100, 114]))
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
            match=re.escape(
                f'Invalid alignment method "{bad_method}". '
                'Must be one of ["CoM" or "MaxImage"]',
            ),
        ):
            stack.tilt_align(bad_method)  # pyright: ignore[reportArgumentType]


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
            match=re.escape(
                f'Invalid interpolation method "{bad_method}". Must be one of '
                '["linear", "cubic", "nearest", or "none"].',
            ),
        ):
            stack.trans_stack(
                1,
                1,
                1,
                bad_method,  # pyright: ignore[reportArgumentType]
            )


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
