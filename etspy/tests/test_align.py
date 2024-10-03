"""Tests for the alignment features of ETSpy."""

import re
from typing import cast

import numpy as np
import pytest
from hyperspy.misc.utils import DictionaryTreeBrowser as Dtb

import etspy.api as etspy
from etspy import datasets as ds


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


class TestAlignStackRegister:
    """Test alignment using stack reg."""

    def test_register_pc(self):
        stack = ds.get_needle_data()
        tomo_meta = cast(Dtb, stack.metadata.Tomography)
        tomo_meta.shifts = tomo_meta.shifts[0:20]
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
        tomo_meta = cast(Dtb, stack.metadata.Tomography)
        tomo_meta.shifts = tomo_meta.shifts[0:20]
        tomo_meta.tilts = tomo_meta.tilts[0:20]
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
        tomo_meta = cast(Dtb, stack.metadata.Tomography)
        tomo_meta.shifts = tomo_meta.shifts[0:20]
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
        tomo_meta = cast(Dtb, stack.metadata.Tomography)
        tomo_meta.shifts = tomo_meta.shifts[0:20]
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
        tomo_meta = cast(Dtb, stack.metadata.Tomography)
        tomo_meta.shifts = tomo_meta.shifts[0:20]
        bad_method = "WRONG"
        with pytest.raises(
            TypeError,
            match=re.escape(
                f'Invalid registration method "{bad_method}". '
                'Must be one of ["StackReg", "PC", "COM", or "COM-CL"].',
            ),
        ):
            stack.inav[0:20].stack_register(bad_method)


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
        cast(Dtb, reg.metadata.Tomography).tilts = None
        with pytest.raises(
            ValueError,
            match="Tilts are not defined in stack.metadata.Tomography",
        ):
            reg.tilt_align(method="CoM", slices=np.array([64, 128, 192]))

    def test_tilt_align_maximage(self):
        stack = ds.get_needle_data()
        stack = stack.inav[0:5]
        stack.metadata.Tomography.shifts = np.zeros([5, 2])
        reg = stack.stack_register("PC")
        ali = reg.tilt_align(method="MaxImage")
        tilt_axis = ali.metadata.Tomography.tiltaxis
        assert isinstance(tilt_axis, float)

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
            stack.tilt_align(method=bad_method)  # pyright: ignore[reportArgumentType]


class TestAlignOther:
    """Test alignment of a dataset calculated for another."""

    def test_align_to_other(self):
        stack = ds.get_needle_data()
        stack = stack.inav[0:5]
        stack.metadata.Tomography.shifts = np.zeros([5, 2])
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
        stack.metadata.Tomography.shifts = np.zeros([5, 2])
        reg = stack.stack_register("PC", crop=True)
        reg2 = reg.align_other(stack)
        diff = reg.data - reg2.data
        assert diff.sum() == 0.0

    def test_align_to_other_with_xshift(self):
        stack = ds.get_needle_data()
        stack = stack.inav[0:5]
        stack.metadata.Tomography.shifts = np.zeros([5, 2])
        stack2 = stack.deepcopy()
        reg = stack.stack_register("PC")
        reg = reg.trans_stack(xshift=10, yshift=5, angle=2)
        reg2 = reg.align_other(stack2)
        diff = reg.data - reg2.data
        assert diff.sum() == 0.0
