"""Tests for the projection matching alignment sub-module of ETSpy."""

from typing import ClassVar, cast
from unittest.mock import patch

import astra
import numpy as np
import pytest
from hyperspy.signals import Signal1D, Signal2D
from scipy import ndimage
from scipy.stats import pearsonr

import etspy.api as etspy
from etspy import datasets as ds
from etspy import projmatch
from etspy import simulation as sim

from . import count_astra_objects

RMSE_THRESH = 0.1
CORR_COEFF = 0.95
NOISE_THRESH = 0.1
AXIS_OFFSET_THRESH = 0.015
ALIGNED_RMS_THRESH = 0.02


@pytest.fixture(scope="class")
def aligned_simulation():
    """Create a simluated misaligned stack with kwown shifts for use with tests."""
    np.random.seed(42)

    # Create the simulation object
    obj = np.zeros([100, 100])
    obj[70:80, 70:80] = 10
    obj[30:65, 30:65] = 8
    obj[15:25, 15:25] = 5
    obj = ndimage.gaussian_filter(obj, sigma=1)

    # Create the simulated tilt series
    sino = sim.create_model_tilt_series(
        obj[np.newaxis, :, :],
        angles=np.linspace(0, 180, 90),
    )
    shifted = sim.misalign_stack(sino, min_shift=-2, max_shift=2, y_only=True)

    parameters = {
        "levels": [4, 2, 1],
        "iterations": 50,
        "minstep": 5e-2,
        "relax": 0.2,
        "recon_algorithm": "SIRT",
        "recon_iterations": 50,
    }

    # Check for CUDA capability
    use_cuda = astra.use_cuda()

    # Calculate the shifts using projection matching
    pm = projmatch.ProjMatch(shifted, nslice=0, cuda=use_cuda, params=parameters)
    pm.calculate_shifts(show_progressbar=False)
    ali = pm.apply_shifts()

    return {
        "known_shifts": shifted.shifts.data[:, 0],
        "calculated_shifts": -pm.total_shifts[:, 0],
        "aligned": ali,
        "stack": shifted,
    }


@pytest.mark.usefixtures("aligned_simulation")
class TestProjmatchAlignment:
    """Test projection matching alignment."""

    def test_tracking_precision(self, aligned_simulation):
        """Assert that relative frame-to-frame tracking error is sub-pixel."""
        known = aligned_simulation["known_shifts"]
        calculated = aligned_simulation["calculated_shifts"]

        delta_known = np.diff(known)
        delta_calculated = np.diff(calculated)

        rmse_deltas = np.sqrt(np.mean((delta_calculated - delta_known) ** 2))

        assert rmse_deltas < RMSE_THRESH, (
            f"Shifts exceed RMSE threshold: {rmse_deltas:.4f}"
        )

    def test_trend_correlation(self, aligned_simulation):
        """Assert that calculated shifts correlate with known shifts."""
        known = aligned_simulation["known_shifts"]
        calculated = aligned_simulation["calculated_shifts"]

        r_value, _ = pearsonr(known, calculated)
        r_value = cast("float", r_value)
        assert r_value > CORR_COEFF, (
            f"Tracking trend correlation is too low: {r_value:.4f}"
        )

    def test_shift_application(self, aligned_simulation):
        stack = aligned_simulation["stack"]
        ali = aligned_simulation["aligned"]
        known = aligned_simulation["known_shifts"]
        calculated = aligned_simulation["calculated_shifts"]

        assert isinstance(ali, etspy.TomoStack)
        assert ali.axes_manager.signal_shape == stack.axes_manager.signal_shape
        assert ali.axes_manager.navigation_shape == stack.axes_manager.navigation_shape
        np.testing.assert_allclose(
            ali.shifts.data[:, 0],
            known - calculated,
            atol=1e-2,
            err_msg="Shifts in aligned stack do not match expected values",
        )


class TestAstraObjectCleanup:
    """Test that projection matching frees the ASTRA objects it creates."""

    pytestmark = pytest.mark.usefixtures("astra_sentinel")

    def test_calculate_shifts_frees_astra_objects(self):
        stack = etspy.TomoStack(
            np.random.default_rng(0).random((15, 32, 1), np.float32),
            tilts=np.linspace(-70, 70, 15),
        )
        before = count_astra_objects()
        pm = projmatch.ProjMatch(
            stack,
            cuda=False,
            params={"levels": [2, 1], "iterations": 3},
        )
        pm.calculate_shifts()
        assert count_astra_objects() == before


class TestProjMatchInit:
    """Test selection of the sinogram used for projection matching."""

    @pytest.mark.parametrize(
        ("ntilts", "nslice", "expected"),
        [(2, None, 1), (5, 0, 0)],
    )
    def test_sinogram_selection(self, ntilts, nslice, expected):
        data = np.random.default_rng(0).random((ntilts, 16, 3), np.float32)
        stack = etspy.TomoStack(data, tilts=np.linspace(-60, 60, ntilts))
        pm = projmatch.ProjMatch(stack, nslice=nslice, cuda=False)
        np.testing.assert_array_equal(pm.sino, data[:, :, expected])

    def test_single_sinogram_stack(self):
        data = np.random.default_rng(0).random((5, 16, 1), np.float32)
        stack = etspy.TomoStack(data, tilts=np.linspace(-60, 60, 5))
        pm = projmatch.ProjMatch(stack, cuda=False)
        np.testing.assert_array_equal(pm.sino, data[:, :, 0])


class TestBackgroundFill:
    """Test that projection matching shifts fill with the background level."""

    def test_shift_sinogram_fills_with_background(self):
        sino = np.full((3, 40), 5.0)
        sino[:, 15:25] = 9.0
        shifted = projmatch.projmatch.shift_sinogram(sino, np.array([3.0, -3.0, 0.0]))
        np.testing.assert_allclose(shifted[0, :3], 5.0, atol=1e-6)
        np.testing.assert_allclose(shifted[1, -3:], 5.0, atol=1e-6)
        np.testing.assert_allclose(shifted[0, 18:28], 9.0, atol=1e-6)
        np.testing.assert_allclose(shifted[2], sino[2])

    def test_shift_sinogram_per_projection_background(self):
        sino = np.array([[1.0] * 20, [2.0] * 20])
        shifted = projmatch.projmatch.shift_sinogram(
            sino,
            np.array([4.0, 4.0]),
            background=np.array([1.0, 2.0]),
        )
        np.testing.assert_allclose(shifted, sino, atol=1e-6)

    def test_apply_shifts_fills_with_background(self):
        data = np.full((4, 32, 8), 7.0, np.float32)
        data[:, 12:20, :] = 10.0
        stack = etspy.TomoStack(data, tilts=np.linspace(-30, 30, 4))
        pm = projmatch.ProjMatch(stack, params={"levels": [1], "iterations": 1})
        pm.total_shifts = np.array([3.0, -3.0, 2.0, 0.0])
        ali = pm.apply_shifts()
        np.testing.assert_allclose(ali.data[0, :3], 7.0, atol=1e-4)
        np.testing.assert_allclose(ali.data[1, -3:], 7.0, atol=1e-4)
        # a 1D array of shifts gives the Y-shifts
        np.testing.assert_allclose(ali.shifts.data[:, 0], pm.total_shifts)
        np.testing.assert_allclose(ali.shifts.data[:, 1], 0)

    @pytest.mark.parametrize("offset", [0.2, -0.2])
    def test_alignment_with_background_offset(self, offset):
        # with zero fill, an offset background made the alignment fail (errors
        # of 2-3 px rms for shifts within +/-4 px)
        small = ds.get_catalyst_data().inav[::2].rebin(scale=(1, 2, 2))
        small.data = small.data / small.data.max()
        np.random.seed(3)
        stack = sim.misalign_stack(small, min_shift=-4, max_shift=4, y_only=True)
        applied = stack.shifts.data[:, 0].copy()
        stack.data = stack.data + offset
        pm = projmatch.ProjMatch(stack, nslice=150)
        pm.calculate_shifts()
        error = pm.total_shifts[:, 0] + applied
        # shifts a + b cos(theta) + c sin(theta) only translate the object
        thetas = np.deg2rad(stack.tilts.data.squeeze())
        basis = np.array([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)]).T
        error = error - basis @ np.linalg.lstsq(basis, error)[0]
        assert np.sqrt((error**2).mean()) < 1.0


@pytest.fixture(scope="module")
def small_misaligned_stack():
    """Small misaligned catalyst tilt series (45 tilts, 150 x 150 pixels)."""
    small = ds.get_catalyst_data().inav[::2].rebin(scale=(1, 4, 4))
    np.random.seed(3)
    return sim.misalign_stack(small, min_shift=-2, max_shift=2, y_only=True)


class TestAlignedInput:
    """Projection matching of an already aligned series."""

    def test_no_axis_offset_from_coarse_levels(self):
        obj = np.zeros([96, 96])
        obj[60:70, 58:72] = 10
        obj[28:56, 30:52] = 8
        obj[16:24, 62:74] = 5
        obj = ndimage.gaussian_filter(obj, sigma=1)
        stack = sim.create_model_tilt_series(
            obj[np.newaxis, :, :],
            angles=np.arange(-90, 90, 2.0),
        )
        pm = projmatch.ProjMatch(stack, nslice=0, params={"levels": [4, 2, 1]})
        pm.calculate_shifts(show_progressbar=False)
        thetas = np.deg2rad(stack.tilts.data.squeeze())
        basis = np.array([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)]).T
        coeffs = np.linalg.lstsq(basis, pm.total_shifts[:, 0])[0]
        # the constant term is the tilt axis offset: determinable over 180 deg
        assert abs(coeffs[0]) < AXIS_OFFSET_THRESH
        rest = pm.total_shifts[:, 0] - basis @ coeffs
        assert np.sqrt((rest**2).mean()) < ALIGNED_RMS_THRESH


class TestHistory:
    """Test the record of the projection matching iterations."""

    params: ClassVar[dict] = {"levels": [2, 1], "iterations": 5, "minstep": 1e-6}

    def test_history_off_by_default(self, small_misaligned_stack):
        pm = projmatch.ProjMatch(small_misaligned_stack, nslice=75, params=self.params)
        pm.calculate_shifts()
        assert pm.sino_update is None
        assert pm.rec_update is None
        assert len(pm.error) == 2  # noqa: PLR2004
        assert all(isinstance(e, Signal1D) for e in pm.error)
        assert all(e.data.shape == (5,) for e in pm.error)
        assert pm.shift_update.data.shape == (2, 45, 2)
        np.testing.assert_array_equal(pm.shift_update.data[-1], pm.total_shifts)

    def test_history_on(self, small_misaligned_stack):
        params = {**self.params, "history": True}
        pm = projmatch.ProjMatch(small_misaligned_stack, nslice=75, params=params)
        pm.calculate_shifts()
        off = projmatch.ProjMatch(
            small_misaligned_stack,
            nslice=75,
            params=self.params,
        )
        off.calculate_shifts()
        np.testing.assert_array_equal(pm.total_shifts, off.total_shifts)
        assert pm.sino_update is not None
        assert pm.rec_update is not None
        for level in range(2):
            assert isinstance(pm.sino_update[level], Signal2D)
            assert isinstance(pm.rec_update[level], Signal2D)
            niter, nangles, ny = pm.sino_update[level].data.shape
            assert (niter, nangles) == (5, 45)
            assert pm.rec_update[level].data.shape == (5, ny, ny)
        assert pm.sino_update[1].data.shape[2] == 150  # noqa: PLR2004

    def test_history_reset_by_second_call(self, small_misaligned_stack):
        pm = projmatch.ProjMatch(small_misaligned_stack, nslice=75, params=self.params)
        pm.calculate_shifts()
        first = pm.total_shifts.copy()
        pm.calculate_shifts()
        # the history is of the second call, which continues from the first
        assert all(e.data.shape == (5,) for e in pm.error)
        assert pm.shift_update.data.shape == (2, 45, 2)
        assert not np.array_equal(pm.total_shifts, first)

    def test_astra_objects_freed_on_error(self, small_misaligned_stack):
        pm = projmatch.ProjMatch(small_misaligned_stack, nslice=75, params=self.params)
        before = count_astra_objects()
        with (
            patch.object(pm, "reconstruct", side_effect=RuntimeError("failed")),
            pytest.raises(RuntimeError, match="failed"),
        ):
            pm.calculate_shifts()
        assert count_astra_objects() == before


def _shift_errors(shifts: np.ndarray, truth: np.ndarray, tilts: np.ndarray):
    """RMS errors of (Y, X) shifts, without the components that cannot be found.

    Y-shifts a + b cos(theta) + c sin(theta) only translate the reconstruction, and
    a constant X-shift only translates it along the tilt axis.
    """
    thetas = np.deg2rad(tilts)
    basis = np.array([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)]).T
    error_y = shifts[:, 0] - truth[:, 0]
    error_y = error_y - basis @ np.linalg.lstsq(basis, error_y, rcond=None)[0]
    error_x = shifts[:, 1] - truth[:, 1]
    error_x = error_x - np.median(error_x)
    return np.sqrt(np.mean(error_y**2)), np.sqrt(np.mean(error_x**2))


@pytest.fixture(scope="module")
def xy_series():
    """Small analytic tilt series misaligned in both X and Y."""
    return sim.create_alignment_test_series(
        shape=(64, 48),
        tilts=np.arange(-60, 61, 4),
        max_shift=2.0,
        drift=1.0,
        tilt_axis=0.0,
        blur=0.7,
        features=(12, 5, 4, 3),
        seed=5,
    )


@pytest.fixture(scope="module")
def xy_alignment(xy_series):
    """Projection matching of the X/Y-misaligned series, using all slices."""
    pm = projmatch.ProjMatch(
        xy_series.misaligned,
        params={"levels": [4, 2, 1], "recon_algorithm": "SIRT", "recon_iterations": 20},
    )
    pm.calculate_shifts()
    return pm


class TestMultiSliceAlignment:
    """Test the alignment using several slices, in both X and Y."""

    def test_recovers_x_and_y_shifts(self, xy_series, xy_alignment):
        tilts = xy_series.misaligned.tilts.data.squeeze()
        before = _shift_errors(np.zeros_like(xy_series.shifts), xy_series.shifts, tilts)
        after = _shift_errors(xy_alignment.total_shifts, xy_series.shifts, tilts)
        assert min(before) > 1.0
        # sub-pixel in both directions (about 0.02 px when measured)
        assert after[0] < 0.1  # noqa: PLR2004
        assert after[1] < 0.1  # noqa: PLR2004

    def test_better_than_single_slice(self, xy_series, xy_alignment):
        # a single slice cannot follow the X-shifts, which change its content
        single = projmatch.ProjMatch(
            xy_series.misaligned,
            nslice=24,
            params={"levels": [4, 2, 1]},
        )
        single.calculate_shifts()
        tilts = xy_series.misaligned.tilts.data.squeeze()
        error_single = _shift_errors(single.total_shifts, xy_series.shifts, tilts)[0]
        error_multi = _shift_errors(xy_alignment.total_shifts, xy_series.shifts, tilts)
        assert error_multi[0] < 0.25 * error_single
        np.testing.assert_array_equal(single.total_shifts[:, 1], 0)

    def test_history_shapes(self, xy_alignment):
        nangles = xy_alignment.nangles
        assert xy_alignment.shift_update.data.shape == (3, nangles, 2)
        assert len(xy_alignment.projection_error) == 3  # noqa: PLR2004
        for errors, projection_errors in zip(
            xy_alignment.error,
            xy_alignment.projection_error,
            strict=True,
        ):
            assert isinstance(projection_errors, Signal1D)
            assert projection_errors.data.shape == (len(errors.data), nangles)
        assert xy_alignment.final_projection_error().shape == (nangles,)

    def test_apply_shifts(self, xy_series):
        # applying the known shifts recovers the aligned series
        pm = projmatch.ProjMatch(xy_series.misaligned)
        pm.total_shifts = xy_series.shifts.copy()
        ali = pm.apply_shifts()
        np.testing.assert_allclose(ali.shifts.data, xy_series.shifts)
        inner = (slice(None), slice(8, -8), slice(8, -8))
        ideal = xy_series.aligned.data[inner] - 0.2
        error = ali.data[inner] - 0.2 - ideal
        assert np.sqrt(np.mean(error**2)) < 0.1 * np.sqrt(np.mean(ideal**2))

    def test_threads_give_identical_results(self, xy_series):
        params = {"levels": [2], "iterations": 3}
        one = projmatch.ProjMatch(xy_series.misaligned, params={**params, "ncores": 1})
        one.calculate_shifts()
        three = projmatch.ProjMatch(
            xy_series.misaligned,
            params={**params, "ncores": 3},
        )
        three.calculate_shifts()
        np.testing.assert_array_equal(one.total_shifts, three.total_shifts)

    def test_restricted_slices(self, xy_series):
        pm = projmatch.ProjMatch(
            xy_series.misaligned,
            params={"slices": slice(8, 40), "levels": [2], "iterations": 3},
        )
        assert pm.slices == (8, 40)
        pm.calculate_shifts()
        assert pm.total_shifts.shape == (xy_series.shifts.shape[0], 2)
        assert np.any(pm.total_shifts[:, 1])


class TestProjectionError:
    """Test the residual of each projection."""

    def test_flags_inconsistent_projection(self, xy_series):
        stack = xy_series.misaligned.deepcopy()
        bad = 10
        # replace one projection by a mirrored copy, inconsistent with the rest
        stack.data[bad] = stack.data[bad, ::-1, ::-1]
        pm = projmatch.ProjMatch(stack, params={"levels": [2, 1], "iterations": 20})
        pm.calculate_shifts()
        error = pm.final_projection_error()
        assert np.argmax(error) == bad
        assert error[bad] > 2 * np.median(error)

    def test_needs_calculate_shifts(self, xy_series):
        pm = projmatch.ProjMatch(xy_series.misaligned)
        with pytest.raises(ValueError, match="Run calculate_shifts first"):
            pm.final_projection_error()


class TestSliceSelection:
    """Test the choice of slices and of the shift directions."""

    @pytest.fixture
    def stack(self):
        data = np.random.default_rng(0).random((5, 16, 12), np.float32)
        return etspy.TomoStack(data, tilts=np.linspace(-60, 60, 5))

    def test_default_uses_all_slices(self, stack):
        pm = projmatch.ProjMatch(stack)
        assert pm.slices == (0, 12)
        assert pm.align_x

    def test_nslice_uses_one_slice_and_y_only(self, stack):
        pm = projmatch.ProjMatch(stack, nslice=3)
        assert pm.slices == (3, 4)
        assert not pm.align_x
        np.testing.assert_array_equal(pm.sino, stack.data[:, :, 3])

    def test_tuple_slices(self, stack):
        pm = projmatch.ProjMatch(stack, params={"slices": (2, 9), "align_x": False})
        assert pm.slices == (2, 9)
        assert not pm.align_x

    @pytest.mark.parametrize(
        ("nslice", "params", "match"),
        [
            (3, {"slices": (0, 5)}, "either nslice or"),
            (12, {}, "nslice must be in the range"),
            (None, {"slices": (5, 5)}, "Invalid slices"),
            (None, {"slices": (0, 13)}, "Invalid slices"),
            (3, {"align_x": True}, "needs more than one slice"),
        ],
    )
    def test_invalid_selection(self, stack, nslice, params, match):
        with pytest.raises(ValueError, match=match):
            projmatch.ProjMatch(stack, nslice=nslice, params=params)

    def test_wrong_number_of_shifts(self, stack):
        pm = projmatch.ProjMatch(stack)
        pm.total_shifts = np.zeros(3)
        with pytest.raises(ValueError, match="Number of shifts"):
            pm.apply_shifts()


class TestDownsampling:
    """Test the helper functions along either axis."""

    def test_blur_and_interpolate_along_x_match_y(self):
        data = np.random.default_rng(1).random((3, 20, 30))
        along_x = projmatch.projmatch.interpolate_ft(
            projmatch.projmatch.blur_convolve(data, 2, axis=2),
            2,
            axis=2,
        )
        transposed = np.swapaxes(data, 1, 2)
        along_y = projmatch.projmatch.interpolate_ft(
            projmatch.projmatch.blur_convolve(transposed, 2, axis=1),
            2,
            axis=1,
        )
        np.testing.assert_allclose(along_x, np.swapaxes(along_y, 1, 2))

    @pytest.mark.parametrize("n", [100, 111, 112])
    @pytest.mark.parametrize("factor", [2, 4, 8])
    def test_binned_positions(self, n, factor):
        pos = projmatch.projmatch.binned_positions(n, factor)
        assert len(pos) == n // factor
        np.testing.assert_allclose(np.diff(pos), factor)
        np.testing.assert_allclose(pos.mean(), (n - 1) / 2)

    @pytest.mark.parametrize("n", [100, 111, 112])
    @pytest.mark.parametrize("factor", [2, 4, 8])
    @pytest.mark.parametrize("offset", [0.0, -15.2, 9.7])
    def test_downsampling_keeps_centre_and_pixel_size(self, n, factor, offset):
        # a centred object stays centred and an off-centre one keeps its distance
        # from the centre in input pixels: otherwise the coarse levels of projection
        # matching see a tilt axis offset (formerly about -0.5 px at every level)
        y = np.arange(n)
        line = np.exp(-0.5 * ((y - (n - 1) / 2 - offset) / 3.0) ** 2)
        data = line[np.newaxis, :, np.newaxis]
        binned = projmatch.projmatch.interpolate_ft(
            projmatch.projmatch.blur_convolve(data, factor, axis=1),
            factor,
            axis=1,
        )[0, :, 0]
        nb = len(binned)
        centroid = (binned * np.arange(nb)).sum() / binned.sum()
        assert (centroid - (nb - 1) / 2) * factor == pytest.approx(offset, abs=0.05)
        # average intensity preserved
        assert binned.sum() * factor == pytest.approx(line.sum(), rel=0.02)

    def test_gradient_of_ramp(self):
        ramp = np.tile(np.arange(32.0), (2, 4, 1))
        grad = projmatch.projmatch._gradient(ramp, axis=2)  # noqa: SLF001
        np.testing.assert_allclose(grad, 1.0)
