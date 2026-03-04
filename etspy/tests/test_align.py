"""Tests for the alignment features of ETSpy."""

import re
import sys
from importlib import reload
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import pytest

import etspy.api as etspy
from etspy import datasets as ds

if TYPE_CHECKING:
    from hyperspy.misc.utils import DictionaryTreeBrowser as Dtb


# Check if CUDA capable GPU is available
cupy_in_test_env = True
try:
    import cupy as cp

    has_gpu = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cupy_in_test_env = False


@pytest.fixture(scope="module")
def short_stack():
    """Load stack from test data and truncate to 5 images."""
    return ds.get_needle_data().inav[0:5]


@pytest.fixture(scope="module")
def full_stack():
    """Load stack from test data."""
    return ds.get_needle_data()


@pytest.fixture(scope="module")
def aligned_short_stack():
    """Create truncated and spatially registered stack from test data."""
    s = ds.get_needle_data().inav[0:5]
    s = s.stack_register("PC")
    return s


@pytest.fixture(scope="module")
def aligned_full_stack():
    """Create full spatially registered stack from test data."""
    s = ds.get_needle_data()
    s = s.stack_register("PC")
    return s


class TestAlignFunctions:
    """Test alignment functions."""

    def test_apply_shifts_bad_shape(self, short_stack):
        shifts = np.zeros(10)
        with pytest.raises(
            ValueError,
            match=r"Number of shifts \(\d+\) is not consistent with number of images "
            r"in the stack \(\d+\)",
        ):
            etspy.align.apply_shifts(short_stack, shifts)

    def test_pad_line(self, short_stack):
        claligner = etspy.align.CommonLineAligner(short_stack)
        line = np.zeros(100)
        padding = 200
        padded = claligner.pad_line(line, padding)
        assert padded.shape[0] == padding

    def test_pad_line_uneven_line(self, short_stack):
        claligner = etspy.align.CommonLineAligner(short_stack)
        line = np.zeros(101)
        padding = 200
        padded = claligner.pad_line(line, padding)
        assert padded.shape[0] == padding

    def test_pad_line_uneven_padded(self, short_stack):
        claligner = etspy.align.CommonLineAligner(short_stack)
        line = np.zeros(100)
        padding = 201
        padded = claligner.pad_line(line, padding)
        assert padded.shape[0] == padding

    def test_pad_line_uneven_both(self, short_stack):
        claligner = etspy.align.CommonLineAligner(short_stack)
        line = np.zeros(101)
        padding = 201
        padded = claligner.pad_line(line, padding)
        assert padded.shape[0] == padding

    def test_calc_shifts_cl_no_index(self, full_stack):
        claligner = etspy.align.CommonLineAligner(full_stack)
        shifts = claligner.calc_shifts_cl(None, 0.05, 8)
        assert isinstance(shifts, np.ndarray)
        assert shifts.shape == (77,)

    def test_calc_shifts_com_with_xrange(self, full_stack):
        comaligner = etspy.align.CoMAligner(full_stack)
        shifts_def = comaligner.calculate_shifts()
        comaligner = etspy.align.CoMAligner(full_stack, xrange=(31, 225))
        shifts = comaligner.calculate_shifts()
        assert isinstance(shifts, np.ndarray)
        assert np.all(shifts == shifts_def)

    def test_calc_shifts_stackreg_no_start(self, full_stack):
        sraligner = etspy.align.StackRegAligner(
            full_stack,
            start=None,
            show_progressbar=False,
        )
        shifts = sraligner.calculate_shifts()
        assert isinstance(shifts, np.ndarray)
        assert shifts.shape == (77, 2)

    def test_tilt_com_no_slices_yes_nslices_30_perc(self, full_stack, caplog):
        com_tilt_aligner = etspy.align.TiltCOMAligner(
            full_stack,
            slices=None,
            nslices=100,
        )
        _ = com_tilt_aligner.align_tilt_axis()
        assert (
            "nslices is greater than 30% of number of x pixels. "
            "Using 76 slices instead."
        ) in caplog.text

    def test_tilt_com_no_slices_yes_nslices_too_big(self, full_stack):
        com_tilt_aligner = etspy.align.TiltCOMAligner(
            full_stack,
            slices=None,
            nslices=300,
        )
        with pytest.raises(
            ValueError,
            match=r"nslices is greater than the X-dimension of the data\.",
        ):
            com_tilt_aligner.align_tilt_axis()

    def test_tilt_com_nx_threshold_error(self, full_stack):
        com_tilt_aligner = etspy.align.TiltCOMAligner(
            full_stack.isig[:2, :],
            slices=None,
            nslices=100,
        )
        with pytest.raises(
            ValueError,
            match=(
                r"Dataset is only 2 pixels in x dimension. This method cannot be used."
            ),
        ):
            com_tilt_aligner.align_tilt_axis()

    def test_calc_shifts_com_cl_res_error(self, full_stack):
        claligner = etspy.align.CommonLineAligner(
            full_stack,
            com_ref_index=30,
            cl_resolution=0.9,
        )
        with pytest.raises(ValueError, match=(r"Resolution should be less than 0.5")):
            claligner.calculate_shifts()


@pytest.mark.skipif(not cupy_in_test_env, reason="cupy not available")
class TestCUDAAlignFunctions:
    """Test alignment functions using CUDA functionality."""

    def test_stack_reg_pc_cuda(self, short_stack):
        short_stack.stack_register("PC", cuda=True)

    def test_no_cupy(self):
        assert etspy.align.has_cupy
        with patch.dict(sys.modules, {"cupy": None}):
            reload(sys.modules["etspy.align"])
            assert not etspy.align.has_cupy
        reload(sys.modules["etspy.align"])
        assert etspy.align.has_cupy

    def test_cuda_cpu_consistency(self, short_stack):
        reg_cuda = short_stack.stack_register("PC", cuda=True)
        shifts_cuda = reg_cuda.shifts.data
        reg_cpu = short_stack.stack_register("PC", cuda=False)
        shifts_cpu = reg_cpu.shifts.data
        assert np.abs(shifts_cuda[0:5] - shifts_cpu[0:5]).sum() < 1.0


class TestAlignStackRegister:
    """Test alignment using stack reg."""

    @pytest.mark.parametrize(
        "method",
        [
            "PC",
            "StackReg",
            "COM",
            "COM-CL",
        ],
    )
    def test_register_methods(self, short_stack, method):
        reg = short_stack.inav[0:20].stack_register(method)
        assert isinstance(reg, etspy.TomoStack)
        assert reg.axes_manager.signal_shape == short_stack.axes_manager.signal_shape
        assert (
            reg.axes_manager.navigation_shape
            == short_stack.axes_manager.navigation_shape
        )

    def test_register_unknown_method(self, short_stack):
        bad_method = "WRONG"
        with pytest.raises(
            TypeError,
            match=re.escape(
                f'Invalid registration method "{bad_method}". '
                'Must be one of ["StackReg", "PC", "COM", or "COM-CL"].',
            ),
        ):
            short_stack.stack_register(bad_method)  # type: ignore


class TestTiltAlign:
    """Test tilt alignment functions."""

    def test_tilt_align_com(self, aligned_full_stack):
        com_tilt_aligner = etspy.align.TiltCOMAligner(
            aligned_full_stack,
            slices=np.array([32, 64, 96, 128, 160]),
        )
        ali = com_tilt_aligner.align_tilt_axis()
        tilt_axis = cast("Dtb", ali.metadata.Tomography).tiltaxis
        assert tilt_axis == pytest.approx(-2.7, abs=0.5)

    def test_tilt_align_com_no_locs(self, aligned_full_stack):
        com_tilt_aligner = etspy.align.TiltCOMAligner(
            aligned_full_stack,
            slices=None,
            nslices=None,
        )
        ali = com_tilt_aligner.align_tilt_axis()
        tilt_axis = cast("Dtb", ali.metadata.Tomography).tiltaxis
        assert tilt_axis == pytest.approx(-3.2, abs=0.5)

    def test_tilt_align_com_no_tilts(self, aligned_full_stack):
        del aligned_full_stack.tilts
        with pytest.raises(
            ValueError,
            match=r"Tilts are not defined in stack.tilts \(values were all zeros\). "
            r"Please set tilt values before alignment.",
        ):
            aligned_full_stack.tilt_align(method="CoM", slices=np.array([64, 128, 192]))

    def test_tilt_align_maximage(self, aligned_full_stack):
        assert aligned_full_stack.metadata.get_item("Tomography.tiltaxis") == 0
        maximage_tilt_aligner = etspy.align.TiltMaxImageAligner(
            aligned_full_stack,
        )
        ali = maximage_tilt_aligner.align_tilt_axis()
        tilt_axis = ali.metadata.get_item("Tomography.tiltaxis")
        assert isinstance(tilt_axis, float)
        assert round(tilt_axis, 1) == pytest.approx(-2.3, rel=1e-1)
        # shifts should be what they were before tilt_align:
        assert np.all(ali.shifts.data == aligned_full_stack.shifts.data)

    # @pytest.mark.mpl_image_compare(remove_text=True)
    def test_tilt_align_maximage_plot_results(self, aligned_short_stack):
        maximage_tilt_aligner = etspy.align.TiltMaxImageAligner(
            aligned_short_stack,
            plot_results=True,
        )
        _ = maximage_tilt_aligner.align_tilt_axis()

    def test_tilt_align_maximage_also_shift(self, aligned_full_stack):
        assert aligned_full_stack.metadata.get_item("Tomography.tiltaxis") == 0
        maximage_tilt_aligner = etspy.align.TiltMaxImageAligner(
            aligned_full_stack, also_shift=True
        )
        ali = maximage_tilt_aligner.align_tilt_axis()
        tilt_axis = ali.metadata.get_item("Tomography.tiltaxis")
        assert isinstance(tilt_axis, float)
        assert round(tilt_axis, 1) == pytest.approx(-2.3, rel=1e-1)
        # also_shift should result in a yshift for the aligned stack
        assert aligned_full_stack.metadata.get_item("Tomography.yshift") == 0
        assert ali.metadata.get_item("Tomography.yshift") == pytest.approx(
            2.0,
            rel=1e-1,
        )

    def test_tilt_align_unknown_method(self, full_stack):
        bad_method = "WRONG"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid alignment method "{bad_method}". '
                'Must be one of ["CoM" or "MaxImage"].',
            ),
        ):
            full_stack.tilt_align(method=bad_method)  # type: ignore


class TestAlignOther:
    """Test alignment of a dataset calculated for another."""

    def test_align_to_other(self, short_stack):
        original_stack = short_stack.deepcopy()
        reg = short_stack.stack_register("PC")
        reg2 = reg.align_other(original_stack)
        diff = reg.data - reg2.data
        assert diff.sum() == 0.0

    def test_align_to_other_no_alignment(self, short_stack):
        original_stack = short_stack.deepcopy()
        reg = short_stack.deepcopy()
        with pytest.raises(
            ValueError,
            match="No transformations have been applied to this stack",
        ):
            reg.align_other(original_stack)

    def test_align_to_other_with_crop(self, short_stack):
        reg = short_stack.stack_register("PC", crop=True)
        reg2 = reg.align_other(short_stack)
        diff = reg.data - reg2.data
        assert diff.sum() == 0.0

    def test_align_to_other_with_xshift(self, short_stack):
        original_stack = short_stack.deepcopy()
        reg = original_stack.stack_register("PC")
        reg = cast("etspy.TomoStack", reg.trans_stack(xshift=10, yshift=5, angle=2))
        reg2 = reg.align_other(original_stack)
        diff = reg.data - reg2.data
        assert diff.sum() == 0.0
