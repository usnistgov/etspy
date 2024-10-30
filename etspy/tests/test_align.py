"""Tests for the alignment features of ETSpy."""

import re
import sys
from importlib import reload
from importlib.util import find_spec
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
from hyperspy.misc.utils import DictionaryTreeBrowser as Dtb

import etspy.api as etspy
from etspy import datasets as ds

cupy_in_test_env = find_spec("cupy") is not None

class TestAlignFunctions:
    """Test alignment functions."""

    def test_apply_shifts_bad_shape(self):
        stack = ds.get_needle_data()
        stack = stack.inav[0:5]
        shifts = np.zeros(10)
        with pytest.raises(
            ValueError,
            match=r"Number of shifts \(\d+\) is not consistent with number of images "
            r"in the stack \(\d+\)",
        ):
            etspy.align.apply_shifts(stack, shifts)

    def test_pad_line(self):
        line = np.zeros(100)
        padding = 200
        padded = etspy.align.pad_line(line, padding)
        assert padded.shape[0] == padding

    def test_pad_line_uneven_line(self):
        line = np.zeros(101)
        padding = 200
        padded = etspy.align.pad_line(line, padding)
        assert padded.shape[0] == padding

    def test_pad_line_uneven_padded(self):
        line = np.zeros(100)
        padding = 201
        padded = etspy.align.pad_line(line, padding)
        assert padded.shape[0] == padding

    def test_pad_line_uneven_both(self):
        line = np.zeros(101)
        padding = 201
        padded = etspy.align.pad_line(line, padding)
        assert padded.shape[0] == padding

    def test_calc_shifts_cl_no_index(self):
        stack = ds.get_needle_data()
        shifts = etspy.align.calc_shifts_cl(stack, None, 0.05, 8)
        assert isinstance(shifts, np.ndarray)
        assert shifts.shape == (77,)

    def test_calc_shifts_com_with_xrange(self):
        stack = ds.get_needle_data()
        shifts_def = etspy.align.calculate_shifts_conservation_of_mass(stack)
        shifts = etspy.align.calculate_shifts_conservation_of_mass(
            stack,
            xrange=(31, 225),
        )
        assert isinstance(shifts, np.ndarray)
        assert np.all(shifts == shifts_def)

    def test_calc_shifts_stackreg_no_start(self):
        stack = ds.get_needle_data()
        shifts = etspy.align.calculate_shifts_stackreg(
            stack,
            start=None,
            show_progressbar=False,
        )
        assert isinstance(shifts, np.ndarray)
        assert shifts.shape == (77, 2)

    def test_tilt_com_no_slices_yes_nslices_30_perc(self, caplog):
        stack = ds.get_needle_data()
        etspy.align.tilt_com(stack, slices=None, nslices=100)
        assert ("nslices is greater than 30% of number of x pixels. "
                "Using 76 slices instead.") in caplog.text

    def test_tilt_com_no_slices_yes_nslices_too_big(self):
        stack = ds.get_needle_data()
        with pytest.raises(
            ValueError,
            match="nslices is greater than the X-dimension of the data.",
        ):
            etspy.align.tilt_com(stack, slices=None, nslices=300)

    def test_tilt_com_nx_threshold_error(self):
        stack = ds.get_needle_data()
        stack = stack.isig[:2, :]
        with pytest.raises(
            ValueError,
            match=(
                "Dataset is only 2 pixels in x dimension. "
                "This method cannot be used."
            ),
        ):
            etspy.align.tilt_com(stack)

    def test_calc_shifts_com_cl_res_error(self):
        stack = ds.get_needle_data()
        with pytest.raises(ValueError,
                           match="Resolution should be less than 0.5"):
            etspy.align.calc_shifts_com_cl(
                stack,
                com_ref_index=30,
                cl_resolution=0.9,
            )

@pytest.mark.skipif(not cupy_in_test_env, reason="cupy not available")
class TestCUDAAlignFunctions:
    """Test alignment functions using CUDA functionality."""

    def test_stack_reg_pc_cuda(self):
        stack = ds.get_needle_data()
        stack.stack_register("PC", cuda=True)

    def test_no_cupy(self):
        assert etspy.align.has_cupy
        with patch.dict(sys.modules, {"cupy": None}):
            reload(sys.modules["etspy.align"])
            assert not etspy.align.has_cupy
        reload(sys.modules["etspy.align"])
        assert etspy.align.has_cupy

class TestAlignStackRegister:
    """Test alignment using stack reg."""

    def test_register_pc(self):
        stack = ds.get_needle_data()
        reg = stack.inav[0:20].stack_register("PC")
        assert isinstance(reg, etspy.TomoStack)
        assert (
            reg.axes_manager.signal_shape == stack.inav[0:20].axes_manager.signal_shape
        )
        assert (
            reg.axes_manager.navigation_shape
            == stack.inav[0:20].axes_manager.navigation_shape
        )

    def test_register_com(self):
        stack = ds.get_needle_data()
        reg = stack.inav[0:20].stack_register("COM")
        assert isinstance(reg, etspy.TomoStack)
        assert (
            reg.axes_manager.signal_shape == stack.inav[0:20].axes_manager.signal_shape
        )
        assert (
            reg.axes_manager.navigation_shape
            == stack.inav[0:20].axes_manager.navigation_shape
        )

    def test_register_stackreg(self):
        stack = ds.get_needle_data()
        reg = stack.inav[0:20].stack_register("StackReg")
        assert isinstance(reg, etspy.TomoStack)
        assert (
            reg.axes_manager.signal_shape == stack.inav[0:20].axes_manager.signal_shape
        )
        assert (
            reg.axes_manager.navigation_shape
            == stack.inav[0:20].axes_manager.navigation_shape
        )

    def test_register_com_cl(self):
        stack = ds.get_needle_data()
        reg = stack.inav[0:20].stack_register("COM-CL")
        assert isinstance(reg, etspy.TomoStack)
        assert (
            reg.axes_manager.signal_shape == stack.inav[0:20].axes_manager.signal_shape
        )
        assert (
            reg.axes_manager.navigation_shape
            == stack.inav[0:20].axes_manager.navigation_shape
        )

    def test_register_unknown_method(self):
        stack = ds.get_needle_data()
        bad_method = "WRONG"
        with pytest.raises(
            TypeError,
            match=re.escape(
                f'Invalid registration method "{bad_method}". '
                'Must be one of ["StackReg", "PC", "COM", or "COM-CL"].',
            ),
        ):
            stack.inav[0:20].stack_register(bad_method)  # type: ignore

    def test_align_stack_bad_method(self):
        """
        Test invalid method in align_stack directly.

        Since we can't through TomoStack.stack_register.
        """
        stack = ds.get_needle_data()
        bad_method = "WRONG"
        with pytest.raises(ValueError,
                           match=f"Invalid alignment method {bad_method}"):
            etspy.align.align_stack(
                stack,
                method=bad_method, # pyright: ignore[reportArgumentType]
                start=None,
                show_progressbar=False,
            )

class TestTiltAlign:
    """Test tilt alignment functions."""

    def test_tilt_align_com(self):
        stack = ds.get_needle_data()
        reg = stack.stack_register("PC")
        ali = reg.tilt_align(method="CoM", slices=np.array([64, 128, 192]))
        tilt_axis = cast(Dtb, ali.metadata.Tomography).tiltaxis
        assert abs(-2.7 - cast(float, tilt_axis)) < 1.0

    def test_tilt_align_com_no_locs(self):
        stack = ds.get_needle_data()
        reg = stack.stack_register("PC")
        ali = reg.tilt_align(method="CoM", slices=None, nslices=None)
        tilt_axis = cast(Dtb, ali.metadata.Tomography).tiltaxis
        assert abs(-2.7 - cast(float, tilt_axis)) < 1.0

    def test_tilt_align_com_no_tilts(self):
        stack = ds.get_needle_data()
        reg = stack.stack_register("PC")
        del reg.tilts
        with pytest.raises(
            ValueError,
            match=r"Tilts are not defined in stack.tilts \(values were all zeros\). "
                  r"Please set tilt values before alignment.",
        ):
            reg.tilt_align(method="CoM", slices=np.array([64, 128, 192]))

    def test_tilt_align_maximage(self):
        stack = ds.get_needle_data()
        stack = stack.inav[0:5]
        reg = stack.stack_register("PC")
        assert reg.metadata.get_item("Tomography.tiltaxis") == 0
        ali = reg.tilt_align(method="MaxImage")
        tilt_axis = ali.metadata.get_item("Tomography.tiltaxis")
        assert isinstance(tilt_axis, float)
        assert tilt_axis == pytest.approx(-0.5)
        # shifts should be what they were before tilt_align:
        assert np.all(ali.shifts.data == reg.shifts.data)

    # @pytest.mark.mpl_image_compare(remove_text=True)
    def test_tilt_align_maximage_plot_results(self):
        stack = ds.get_needle_data()
        stack = stack.inav[0:5]
        reg = stack.stack_register("PC")
        reg.tilt_align(method="MaxImage", plot_results=True)

    def test_tilt_align_maximage_also_shift(self):
        stack = ds.get_needle_data()
        stack = stack.inav[0:5]
        reg = stack.stack_register("PC")
        assert reg.metadata.get_item("Tomography.tiltaxis") == 0
        ali = reg.tilt_align(method="MaxImage", also_shift=True)
        tilt_axis = ali.metadata.get_item("Tomography.tiltaxis")
        assert isinstance(tilt_axis, float)
        assert tilt_axis == pytest.approx(-0.5)
        # also_shift should result in a yshift for the aligned stack
        assert reg.metadata.get_item("Tomography.yshift") == 0
        assert ali.metadata.get_item("Tomography.yshift") == pytest.approx(-19)

    def test_tilt_align_unknown_method(self):
        stack = ds.get_needle_data()
        bad_method = "WRONG"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid alignment method "{bad_method}". '
                'Must be one of ["CoM" or "MaxImage"].',
            ),
        ):
            stack.tilt_align(method=bad_method)  # type: ignore


class TestAlignOther:
    """Test alignment of a dataset calculated for another."""

    def test_align_to_other(self):
        stack = ds.get_needle_data()
        stack = stack.inav[0:5]
        stack2 = stack.deepcopy()
        reg = stack.stack_register("PC")
        reg2 = reg.align_other(stack2)
        diff = reg.data - reg2.data
        assert diff.sum() == 0.0

    def test_align_to_other_no_alignment(self):
        stack = ds.get_needle_data()
        stack = stack.inav[0:5]
        stack2 = stack.deepcopy()
        reg = stack.deepcopy()
        with pytest.raises(
            ValueError,
            match="No transformations have been applied to this stack",
        ):
            reg.align_other(stack2)

    def test_align_to_other_with_crop(self):
        stack = ds.get_needle_data()
        stack = stack.inav[0:5]
        reg = stack.stack_register("PC", crop=True)
        reg2 = reg.align_other(stack)
        diff = reg.data - reg2.data
        assert diff.sum() == 0.0

    def test_align_to_other_with_xshift(self):
        stack = ds.get_needle_data()
        stack = stack.inav[0:5]
        stack2 = stack.deepcopy()
        reg = stack.stack_register("PC")
        reg = cast(etspy.TomoStack, reg.trans_stack(xshift=10, yshift=5, angle=2))
        reg2 = reg.align_other(stack2)
        diff = reg.data - reg2.data
        assert diff.sum() == 0.0
