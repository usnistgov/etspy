"""Test the reconstruction module of ETSpy."""

import re
from typing import cast

import astra
import numpy as np
import pytest

from etspy import datasets as ds
from etspy import recon
from etspy.base import RecStack, TomoStack


@pytest.fixture(scope="module")
def aligned_full_stack():
    """Create full spatially registered stack from test data."""
    s = ds.get_needle_data(aligned=True)
    return s


@pytest.fixture(scope="module")
def aligned_sinogram(aligned_full_stack):
    """Extract single sinogram from aligned full stack."""
    s = aligned_full_stack.isig[120:121, :].deepcopy()
    return s


class TestReconstruction:
    """Test TomoStack reconstruction methods."""

    def test_recon_no_tilts(self, aligned_sinogram):
        aligned_sinogram_no_tilts = aligned_sinogram.deepcopy()
        del aligned_sinogram_no_tilts.tilts
        with pytest.raises(
            ValueError,
            match=r"Tilts are not defined in stack.tilts \(values were all zeros\). "
            r"Please set tilt values before alignment.",
        ):
            aligned_sinogram_no_tilts.reconstruct("FBP")

    def test_recon_single_slice(self, aligned_full_stack, aligned_sinogram):
        tilts = aligned_full_stack.tilts.data.squeeze()
        rec = recon.run(aligned_sinogram.data, tilts, "FBP", cuda=False)
        assert isinstance(aligned_full_stack, TomoStack)
        assert isinstance(rec, np.ndarray)
        data_shape = rec.data.shape
        data_shape = cast("tuple[int, int, int]", data_shape)  # cast for type checking
        assert data_shape[2] == aligned_sinogram.data.shape[1]

    def test_recon_unknown_algorithm(self, aligned_sinogram):
        bad_method = "UNKNOWN"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid reconstruction algorithm "{bad_method}". Must be one of '
                '["FBP", "SIRT", "SART", or "DART"]',
            ),
        ):
            aligned_sinogram.reconstruct(bad_method)  # type: ignore

    def test_recon_fbp_cpu(self, aligned_sinogram):
        rec = aligned_sinogram.reconstruct("FBP", cuda=False)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == aligned_sinogram.data.shape[1]

    def test_recon_fbp_cpu_multicore(self, aligned_sinogram):
        rec = aligned_sinogram.reconstruct("FBP", cuda=False)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == aligned_sinogram.data.shape[1]

    def test_recon_sirt_cpu(self, aligned_sinogram):
        rec = aligned_sinogram.reconstruct(
            "SIRT",
            constrain=True,
            iterations=2,
            thresh=0,
            cuda=False,
        )
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == aligned_sinogram.data.shape[1]

    def test_recon_sart_cpu(self, aligned_sinogram):
        rec = aligned_sinogram.reconstruct(
            "SART",
            constrain=True,
            iterations=2,
            thresh=0,
            cuda=False,
        )
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == aligned_sinogram.data.shape[1]

    def test_recon_dart_cpu(self, aligned_sinogram):
        gray_levels = [
            0.0,
            aligned_sinogram.data.max() / 2,
            aligned_sinogram.data.max(),
        ]
        rec = aligned_sinogram.reconstruct(
            "DART",
            iterations=2,
            cuda=False,
            gray_levels=gray_levels,
            dart_iterations=1,
            ncores=1,
        )
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == aligned_sinogram.data.shape[1]

    def test_recon_dart_cpu_multicore(self, aligned_sinogram):
        gray_levels = [
            0.0,
            aligned_sinogram.data.max() / 2,
            aligned_sinogram.data.max(),
        ]
        rec = aligned_sinogram.reconstruct(
            "DART",
            iterations=2,
            cuda=False,
            gray_levels=gray_levels,
            dart_iterations=1,
            ncores=1,
        )
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == aligned_sinogram.data.shape[1]


class TestReconRun:
    """Test the reconstruction "run" method."""

    def test_run_fbp_no_cuda(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        rec = recon.run(aligned_sinogram.data, tilts, "FBP", cuda=False)
        data_shape = cast("tuple[int, int, int]", rec.data.shape)
        assert data_shape == (
            1,
            aligned_sinogram.data.shape[1],
            aligned_sinogram.data.shape[1],
        )
        assert data_shape[0] == aligned_sinogram.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_fbp_no_cuda_2d_array(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        rec = recon.run(aligned_sinogram.data.squeeze(), tilts, "FBP", cuda=False)
        data_shape = cast("tuple[int, int, int]", rec.data.shape)
        assert data_shape == (
            1,
            aligned_sinogram.data.shape[1],
            aligned_sinogram.data.shape[1],
        )
        assert data_shape[0] == aligned_sinogram.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_sirt_no_cuda(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        rec = recon.run(aligned_sinogram.data, tilts, "SIRT", niterations=2, cuda=False)
        data_shape = cast("tuple[int, int, int]", rec.data.shape)
        assert data_shape == (
            1,
            aligned_sinogram.data.shape[1],
            aligned_sinogram.data.shape[1],
        )
        assert data_shape[0] == aligned_sinogram.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_sart_no_cuda(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        rec = recon.run(aligned_sinogram.data, tilts, "SART", niterations=2, cuda=False)
        data_shape = cast("tuple[int, int, int]", rec.data.shape)
        assert data_shape == (
            1,
            aligned_sinogram.data.shape[1],
            aligned_sinogram.data.shape[1],
        )
        assert data_shape[0] == aligned_sinogram.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_dart_no_cuda(self, aligned_sinogram):
        gray_levels = [
            0.0,
            aligned_sinogram.data.max() / 2,
            aligned_sinogram.data.max(),
        ]
        tilts = aligned_sinogram.tilts.data.squeeze()
        rec = recon.run(
            aligned_sinogram.data,
            tilts,
            "DART",
            niterations=2,
            cuda=False,
            gray_levels=gray_levels,
            dart_iterations=1,
        )
        data_shape = cast("tuple[int, int, int]", rec.data.shape)
        assert data_shape == (
            1,
            aligned_sinogram.data.shape[1],
            aligned_sinogram.data.shape[1],
        )
        assert data_shape[0] == aligned_sinogram.data.shape[2]
        assert isinstance(rec, np.ndarray)

    @pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
    def test_run_dart_no_gray_levels_cuda(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        with pytest.raises(ValueError, match="gray_levels must be provided for DART"):
            recon.run(
                aligned_sinogram.data,
                tilts,
                "DART",
                niterations=2,
                cuda=True,
                gray_levels=None,
                dart_iterations=1,
            )

    def test_run_dart_no_gray_levels_no_cuda(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        with pytest.raises(ValueError, match="gray_levels must be provided for DART"):
            recon.run(
                aligned_sinogram.data,
                tilts,
                "DART",
                niterations=2,
                cuda=False,
                gray_levels=None,
                dart_iterations=1,
            )

    def test_run_dart_no_cuda_no_progress(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        rec = recon.run(
            aligned_sinogram.data,
            tilts,
            "DART",
            niterations=2,
            cuda=False,
            gray_levels=[
                0.0,
                aligned_sinogram.data.max() / 2,
                aligned_sinogram.data.max(),
            ],
            dart_iterations=1,
            show_progressbar=False,
        )
        data_shape = cast("tuple[int, int, int]", rec.data.shape)
        assert data_shape == (
            1,
            aligned_sinogram.data.shape[1],
            aligned_sinogram.data.shape[1],
        )
        assert data_shape[0] == aligned_sinogram.data.shape[2]
        assert isinstance(rec, np.ndarray)


class TestAstraError:
    """Test Astra toolbox errors."""

    def test_astra_sirt_error_cpu(self, aligned_sinogram):
        _, ny, _ = aligned_sinogram.data.shape
        angles = aligned_sinogram.tilts.data.squeeze()
        rec_stack, error = recon.astra_error(
            aligned_sinogram.data[:, :, 0],
            angles,
            iterations=2,
            constrain=True,
            thresh=0,
            cuda=False,
        )
        assert isinstance(error, np.ndarray)
        assert rec_stack.shape == (2, ny, ny)

    def test_astra_sart_error_cpu(self, aligned_sinogram):
        _, ny, _ = aligned_sinogram.data.shape
        angles = aligned_sinogram.tilts.data.squeeze()
        rec_stack, error = recon.astra_error(
            aligned_sinogram.data[:, :, 0],
            angles,
            method="SART",
            iterations=2,
            constrain=True,
            thresh=0,
            cuda=False,
        )
        assert isinstance(error, np.ndarray)
        assert rec_stack.shape == (2, ny, ny)

    def test_astra_error_cpu_bad_dims(self, aligned_sinogram):
        angles = aligned_sinogram.tilts.data.squeeze()
        sino = np.random.rand(3, 5, 10)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Sinogram must be two-dimensional (ntilts, y). "
                "Provided shape was (3, 5, 10).",
            ),
        ):
            recon.astra_error(
                sino,
                angles,
                method="SART",
                iterations=2,
                constrain=True,
                thresh=0,
                cuda=False,
            )

    def test_astra_error_cpu_nangles_mismatch(self, aligned_sinogram):
        angles = aligned_sinogram.tilts.data.squeeze()
        sino = np.random.rand(3, 10)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Number of angles must match size of the first dimension of the "
                "sinogram. [len(angles) was 77; sinogram.shape was (3, 10)] "
                "(77 != 3)",
            ),
        ):
            recon.astra_error(
                sino,
                angles,
                method="SART",
                iterations=2,
                constrain=True,
                thresh=0,
                cuda=False,
            )
