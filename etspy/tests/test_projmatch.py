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
        "calculated_shifts": -pm.total_shifts,
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
        np.testing.assert_allclose(ali.shifts.data[:, 0], pm.total_shifts)

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
        error = pm.total_shifts + applied
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
        assert pm.shift_update.data.shape == (2, 45)
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
        assert pm.shift_update.data.shape == (2, 45)
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
