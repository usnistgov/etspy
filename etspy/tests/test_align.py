"""Tests for the alignment features of ETSpy."""

import re
import sys
from importlib import reload
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import pytest
from scipy import ndimage

import etspy.api as etspy
from etspy import datasets as ds
from etspy import simulation as sim

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


@pytest.fixture(scope="module")
def catalyst_stack():
    """Create full spatially registered stack from test data."""
    s = ds.get_catalyst_data()
    return s


class TestProgressBars:
    """Test that progress bars respect show_progressbar."""

    def test_common_line_no_progressbar(self, short_stack, capsys):
        aligner = etspy.align.CommonLineAligner(short_stack, show_progressbar=False)
        aligner.calculate_shifts()
        assert "it/s" not in capsys.readouterr().err

    def test_common_line_progressbar(self, short_stack, capsys):
        aligner = etspy.align.CommonLineAligner(short_stack, show_progressbar=True)
        aligner.calculate_shifts()
        assert "it/s" in capsys.readouterr().err


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

    @pytest.mark.parametrize("com_ref_index", [0, 2, 4])
    def test_calc_yshifts_relative_to_reference(self, com_ref_index):
        # Gaussian blob displaced perpendicular to the tilt axis by known amounts
        offsets = np.array([-3.0, -1.5, 0.0, 2.0, 4.5])
        yy, xx = np.mgrid[0:64, 0:64]
        data = np.stack(
            [np.exp(-((yy - 32 - d) ** 2 + (xx - 32) ** 2) / 50) for d in offsets],
        ).astype(np.float32)
        stack = etspy.TomoStack(data, tilts=np.linspace(-60, 60, 5))
        claligner = etspy.align.CommonLineAligner(stack, com_ref_index=com_ref_index)
        yshifts = claligner._calc_yshifts(com_ref_index)  # noqa: SLF001
        np.testing.assert_allclose(
            yshifts,
            offsets[com_ref_index] - offsets,
            atol=0.01,
        )


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
        stack_no_tilts = aligned_full_stack.deepcopy()
        del stack_no_tilts.tilts
        with pytest.raises(
            ValueError,
            match=r"Tilts are not defined in stack.tilts \(values were all zeros\). "
            r"Please set tilt values before alignment.",
        ):
            stack_no_tilts.tilt_align(method="CoM", slices=np.array([64, 128, 192]))

    def test_tilt_align_maximage(self, catalyst_stack):
        assert catalyst_stack.metadata.get_item("Tomography.tiltaxis") == 0
        rotated = catalyst_stack.trans_stack(angle=2.3)
        maximage_tilt_aligner = etspy.align.TiltMaxImageAligner(
            rotated,
        )
        ali = maximage_tilt_aligner.align_tilt_axis()
        tilt_axis = ali.metadata.get_item("Tomography.tiltaxis")
        assert isinstance(tilt_axis, float)
        assert round(tilt_axis, 1) == pytest.approx(-2.3, rel=1e-1)
        # shifts should be what they were before tilt_align:
        assert np.all(ali.shifts.data == catalyst_stack.shifts.data)

    # @pytest.mark.mpl_image_compare(remove_text=True)
    def test_tilt_align_maximage_plot_results(self, aligned_short_stack):
        maximage_tilt_aligner = etspy.align.TiltMaxImageAligner(
            aligned_short_stack,
            plot_results=True,
        )
        _ = maximage_tilt_aligner.align_tilt_axis()

    def test_tilt_align_maximage_also_shift(self, catalyst_stack):
        assert catalyst_stack.metadata.get_item("Tomography.tiltaxis") == 0
        assert catalyst_stack.metadata.get_item("Tomography.yshift") == 0
        misaligned = catalyst_stack.trans_stack(angle=2.3, yshift=2)
        maximage_tilt_aligner = etspy.align.TiltMaxImageAligner(
            misaligned,
        )
        ali = maximage_tilt_aligner.align_tilt_axis()
        tilt_axis = ali.metadata.get_item("Tomography.tiltaxis")
        tilt_shift = ali.metadata.get_item("Tomography.yshift")
        assert isinstance(tilt_axis, float)
        assert round(tilt_axis, 1) == pytest.approx(-2.3, rel=1e-1)
        assert round(tilt_shift, 1) == pytest.approx(2.0, rel=1e-1)
        assert round(tilt_axis, 1) == pytest.approx(-2.3, rel=1e-1)
        assert np.all(ali.shifts.data == catalyst_stack.shifts.data)

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

    @staticmethod
    def _assert_close_interior(actual, expected, max_rel, margin=15):
        """Compare away from the edges, relative to the maximum of the data."""
        interior = (slice(None), slice(margin, -margin), slice(margin, -margin))
        diff = np.abs(actual.data[interior] - expected.data[interior])
        assert diff.max() <= max_rel * expected.data.max()

    def test_align_to_other(self, short_stack):
        original_stack = short_stack.deepcopy()
        reg = short_stack.stack_register("PC")
        reg2 = reg.align_other(original_stack)
        assert reg2.data.shape == reg.data.shape
        np.testing.assert_array_equal(reg2.shifts.data, reg.shifts.data)
        # both apply the shifts by Fourier shift (with different padding)
        self._assert_close_interior(reg2, reg, 2e-3)

    def test_align_to_other_no_alignment(self, short_stack):
        original_stack = short_stack.deepcopy()
        reg = short_stack.deepcopy()
        with pytest.raises(
            ValueError,
            match="No transformations have been applied to this stack",
        ):
            reg.align_other(original_stack)

    @staticmethod
    def _stack_with_shifts(shifts):
        data = np.random.default_rng(0).random((len(shifts), 40, 50), np.float32)
        stack = etspy.TomoStack(data, tilts=np.arange(len(shifts), dtype=float))
        stack.shifts = np.array(shifts, dtype=float)
        return stack

    def test_shift_crop_axes(self):
        # shifts[:, 0] are Y-shifts (rows), shifts[:, 1] are X-shifts (columns)
        stack = self._stack_with_shifts(
            [[-2.0, 0.0], [1.0, -6.0], [2.5, 3.0], [3.0, 1.0], [1.0, 1.0]],
        )
        cropped = etspy.align.shift_crop(stack)
        # Y: rows 3 through 40 - 2 = 38; X: columns 3 through 50 - 6 = 44
        assert cropped.data.shape == (5, 35, 41)
        np.testing.assert_array_equal(cropped.data, stack.data[:, 3:38, 3:44])
        assert cropped.metadata.get_item("Tomography.cropped")

    def test_shift_crop_non_negative_shifts(self):
        stack = self._stack_with_shifts(
            [[0.0, 0.0], [1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 1.0]],
        )
        cropped = etspy.align.shift_crop(stack)
        assert cropped.data.shape == (5, 37, 47)
        np.testing.assert_array_equal(cropped.data, stack.data[:, 3:, 3:])

    def test_shift_crop_no_shifts(self):
        stack = self._stack_with_shifts(np.zeros((5, 2)))
        cropped = etspy.align.shift_crop(stack)
        assert cropped.data.shape == stack.data.shape

    def test_shift_crop_too_large(self):
        stack = self._stack_with_shifts(
            [[-25.0, 0.0], [20.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        )
        with pytest.raises(ValueError, match="Shifts are too large"):
            etspy.align.shift_crop(stack)

    def test_align_to_other_with_crop(self, short_stack):
        reg = short_stack.stack_register("PC", crop=True)
        reg2 = reg.align_other(short_stack)
        assert reg2.data.shape == reg.data.shape
        assert reg2.metadata.Tomography.cropped
        self._assert_close_interior(reg2, reg, 2e-3)

    def test_align_to_other_with_xshift(self, short_stack):
        original_stack = short_stack.deepcopy()
        reg = original_stack.stack_register("PC")
        reg = cast(
            "etspy.TomoStack",
            reg.trans_stack(xshift=10, yshift=5, angle=2, interpolation="fourier"),
        )
        reg2 = reg.align_other(original_stack)
        tomo_meta = reg2.metadata.Tomography
        assert (tomo_meta.xshift, tomo_meta.yshift, tomo_meta.tiltaxis) == (10, 5, 2)
        # a single resampling of the original data matches shifting and then
        # rotating (except where the intermediate step lost content at the edges)
        interior = (slice(None), slice(15, -15), slice(15, -15))
        diff = np.abs(reg2.data[interior] - reg.data[interior])
        assert diff.mean() < 0.005 * reg.data[interior].mean()

    @pytest.mark.parametrize("cropped", [False, True])
    def test_align_to_other_single_resampling_is_exact(self, cropped):
        # on smooth data, the single Fourier resampling equals shifting the
        # images, cropping, and rotating/shifting about the (cropped) center
        yy, xx = np.mgrid[:96, :128]
        rng = np.random.default_rng(0)
        images = [
            sum(
                np.exp(
                    -(
                        ((xx - rng.uniform(30, 98)) / 7) ** 2
                        + ((yy - rng.uniform(25, 70)) / 6) ** 2
                    ),
                )
                for _ in range(5)
            )
            for _ in range(6)
        ]
        raw = etspy.TomoStack(
            np.array(images, np.float32),
            tilts=np.linspace(-50, 50, 6),
        )
        shifts = np.column_stack([rng.uniform(-4, 4, 6), rng.uniform(-3, 3, 6)])
        ref = etspy.align.apply_shifts(raw, shifts, "fourier")
        if cropped:
            ref = etspy.align.shift_crop(ref)
        ref = ref.trans_stack(1.5, -3.2, 6.0, interpolation="fourier")
        out = ref.align_other(raw)
        assert out.data.shape == ref.data.shape
        self._assert_close_interior(out, ref, 2e-3, margin=8)


@pytest.fixture(scope="module")
def blob_tilt_series():
    """
    Tilt series of randomly placed blobs, with the tilt axis along x.

    Unlike a specimen that is symmetric about the tilt axis, this contains mass
    away from the axis at all positions along it, so misalignment of the tilt
    axis can be detected from the centers of mass of the projections.
    """
    rng = np.random.default_rng(0)
    nx, nyz = 240, 160
    model = np.zeros((nx, nyz, nyz), np.float32)  # (x, z, y)
    for _ in range(40):
        x = rng.uniform(15, nx - 15)
        radius, angle = rng.uniform(0, 0.3 * nyz), rng.uniform(0, 2 * np.pi)
        z = nyz / 2 + radius * np.cos(angle)
        y = nyz / 2 + radius * np.sin(angle)
        model[int(x), int(z), int(y)] += rng.uniform(0.5, 1.5)
    model = ndimage.gaussian_filter(model, 3)
    return sim.create_model_tilt_series(model, np.arange(-70, 71, 2.0), cuda=False)


def _misalign_tilt_axis(stack, yshift, angle):
    out = stack.deepcopy()
    data = ndimage.shift(stack.data, [0, yshift, 0], order=1)
    out.data = ndimage.rotate(data, angle, axes=(1, 2), reshape=False, order=1)
    return out


class TestTiltCOMAccuracy:
    """Test that the CoM tilt axis alignment recovers known misalignments."""

    @pytest.mark.parametrize(
        ("yshift", "angle"),
        [(8.0, 0.0), (0.0, 3.0), (8.0, 3.0), (-5.0, -2.0), (3.0, -4.0)],
    )
    def test_tilt_com_recovers_misalignment(self, blob_tilt_series, yshift, angle):
        misaligned = _misalign_tilt_axis(blob_tilt_series, yshift, angle)
        aligned = misaligned.tilt_align("CoM", nslices=20)
        tomo_meta = aligned.metadata.Tomography
        assert tomo_meta.yshift == pytest.approx(yshift, abs=0.5)
        assert tomo_meta.tiltaxis == pytest.approx(angle, abs=0.5)

    def test_tilt_com_aligned_input_unchanged(self, blob_tilt_series):
        aligned = blob_tilt_series.tilt_align("CoM", nslices=20)
        tomo_meta = aligned.metadata.Tomography
        assert tomo_meta.yshift == pytest.approx(0, abs=0.05)
        assert tomo_meta.tiltaxis == pytest.approx(0, abs=0.05)

    def test_tilt_com_background_offset(self, blob_tilt_series):
        # a constant background must not change the result (pixels uncovered by
        # the transformation in later iterations are filled with zeros)
        misaligned = _misalign_tilt_axis(blob_tilt_series, 5.0, 3.0)
        misaligned.data = misaligned.data * 1e4  # realistic counts (max ~500)
        offset = misaligned.deepcopy()
        offset.data = offset.data - 30000
        meta = misaligned.tilt_align("CoM", nslices=20).metadata.Tomography
        meta_offset = offset.tilt_align("CoM", nslices=20).metadata.Tomography
        assert meta_offset.yshift == pytest.approx(meta.yshift, abs=0.05)
        assert meta_offset.tiltaxis == pytest.approx(meta.tiltaxis, abs=0.05)

    def test_get_coms_pixel_coordinates(self):
        # mass at row 10 of 64 should have a CoM of 10 - 31.5 pixels
        data = np.zeros((3, 64, 4), np.float32)
        data[:, 10, :] = 1
        stack = etspy.TomoStack(data, tilts=np.array([-10.0, 0.0, 10.0]))
        coms = etspy.align.get_coms(stack, np.array([0, 3]))
        np.testing.assert_allclose(coms, 10 - 31.5)


class TestCoMBackgroundIndependence:
    """Test that center of mass alignment does not depend on the background."""

    def test_get_coms_independent_of_offset(self, blob_tilt_series):
        slices = np.arange(30, 210, 20)
        offset = blob_tilt_series.deepcopy()
        offset.data = offset.data * 1e4 - 30000
        scaled = blob_tilt_series.deepcopy()
        scaled.data = scaled.data * 1e4
        np.testing.assert_allclose(
            etspy.align.get_coms(offset, slices),
            etspy.align.get_coms(scaled, slices),
            atol=1e-3,
        )

    @pytest.mark.parametrize("background", [0.0, 4000.0, -30000.0])
    def test_tilt_com_with_background_and_noise(self, blob_tilt_series, background):
        misaligned = _misalign_tilt_axis(blob_tilt_series, 5.0, 3.0)
        rng = np.random.default_rng(1)
        signal = misaligned.data / misaligned.data.max() * 1000
        misaligned.data = signal + background + rng.normal(0, 20, signal.shape)
        tomo_meta = misaligned.tilt_align("CoM", nslices=20).metadata.Tomography
        assert tomo_meta.yshift == pytest.approx(5.0, abs=1.0)
        assert tomo_meta.tiltaxis == pytest.approx(3.0, abs=0.3)


@pytest.fixture(scope="module")
def misaligned_catalyst():
    """Catalyst tilt series with known random shifts (stored in ``.shifts``)."""
    np.random.seed(1)
    return ds.get_catalyst_data(misalign=True)


def _rms_error(measured, expected):
    """RMS difference, ignoring a constant offset."""
    diff = measured - expected
    return np.sqrt(((diff - diff.mean()) ** 2).mean())


class TestCoMAligner:
    """Test the center of mass stack alignment."""

    def test_yshifts_recover_applied_shifts(self, misaligned_catalyst):
        # the correction is the negative of the applied jitter; a constant offset
        # cannot be determined (it is a shift of the tilt axis)
        shifts = etspy.align.CoMAligner(misaligned_catalyst).calculate_shifts()
        applied = misaligned_catalyst.shifts.data
        assert _rms_error(shifts[:, 0], -applied[:, 0]) < 0.4  # noqa: PLR2004
        assert _rms_error(shifts[:, 1], -applied[:, 1]) < 0.5  # noqa: PLR2004

    def test_xshifts_small_image(self, misaligned_catalyst):
        # an 80 pixel wide image previously gave a negative slice index
        small = misaligned_catalyst.isig[260:340, :].deepcopy()
        shifts = etspy.align.CoMAligner(small).calculate_shifts()
        applied = misaligned_catalyst.shifts.data
        assert _rms_error(shifts[:, 1], -applied[:, 1]) < 0.6  # noqa: PLR2004

    def test_xshifts_too_small_image(self):
        stack = etspy.TomoStack(np.ones([3, 16, 2]), tilts=np.array([-5.0, 0, 5]))
        with pytest.raises(ValueError, match="Not enough pixels along the x-axis"):
            etspy.align.CoMAligner(stack).calculate_shifts()

    @pytest.mark.parametrize("shift", [-5, 5])
    def test_xshifts_full_search_range(self, shift):
        # shifts of both +p and -p pixels can be found
        x = np.arange(100)
        profile = np.exp(-(((x - 40) / 3.0) ** 2)) + np.exp(-(((x - 60) / 5.0) ** 2))
        data = np.tile(profile, (21, 8, 1))
        data[2] = np.roll(data[2], shift, axis=1)
        stack = etspy.TomoStack(data, tilts=np.linspace(-50, 50, 21))
        shifts = etspy.align.CoMAligner(stack, p=5).calculate_shifts()
        expected = np.zeros(21)
        expected[2] = -shift
        np.testing.assert_array_equal(shifts[:, 1], expected)
